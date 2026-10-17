# PROJECT_REPORT.md — MiniOneRec-CAT

**Semantic-ID quantisation and collaborative alignment for generative recommendation**

`Industrial_and_Scientific` · Qwen2.5-0.5B · 3 686 items · 36 259 / 4 532 / 4 533 train/valid/test rows

> **Headline, stated honestly:** collaborative alignment (B2) demonstrably reshaped the
> SID codebook and, at λ=0.10, produced a **positive but not statistically significant**
> point estimate on the ranking metrics. A frequency-aware SFT loss then **did** move the
> exposure distribution substantially and **did** improve every long-tail stratum with
> confidence intervals excluding zero — while **significantly degrading overall NDCG**.
> The project's durable contributions are the **measurement infrastructure and the negative
> results**, not a claimed accuracy win.

---

## 1. Motivation and problem definition

MiniOneRec reframes recommendation as **constrained sequence generation**: each item gets a
**Semantic ID** — a short tuple of discrete tokens such as `<a_236><b_91><c_104>` — and a
language model is trained to emit the next item's SID given the user's history of SIDs.
Decoding is a beam search restricted to a **prefix trie** built from the item catalogue, so
every emitted sequence is a legal item.

That design makes the **quantiser the ceiling of the whole system**. The SID assignment must
simultaneously:

1. keep **semantically similar items** near each other (so a small generation error still
   lands on a plausible item) — this is what the text-embedding RQ-VAE provides;
2. be **collision-free enough** that decoding maps back to a unique item;
3. be **learnable** by a small LM under a fixed token budget.

**Research question:** can we improve the SID by injecting **collaborative signal** — who
co-occurs with whom — so that the codebook reflects behaviour as well as text?

The honest framing matters: the metric is **SID-level**, the catalogue has collisions, and a
SID improvement is only worth something if it survives an end-to-end SFT + generation
evaluation. This project builds that entire chain.

## 2. MiniOneRec upstream vs B0

| | Upstream / P0 | **B0** |
|---|---|---|
| Teacher embedding | Qwen text embedding, `emb-qwen-td.npy` (3 686 × 2 560, float16, row-norm ≈ 92, **not L2-normalised**) | **identical file** (`91a1c875e4b009f3`) |
| RQ-VAE training budget | unknown — **checkpoint never published** | **10 000 optimizer steps**, documented |
| Seed | unknown | **2024 and 7**, both run |
| Reproducibility | **not reproducible** (no quantiser checkpoint anywhere; searched local + AutoDL, 0 hits) | reproducible end-to-end |
| Collision handling | unknown | 20-round Sinkhorn retry, documented |

**★ Why a training-budget trap had to be found the hard way.** `rq/rqvae.sh` passes
`--epochs 10000 --batch_size 20480`, but the argparse block in `rq/rqvae.py:17-47` declares
`epochs=5000, batch_size=2048`. With 3 686 items, a batch of 20 480 means **one optimizer
step per epoch**, so the launcher's 10 000 epochs = **10 000 steps**. Using the parser
defaults gives 2 batches/epoch, so 300 epochs = **600 steps — 16.7× short**.
That first run was preserved as evidence (`runs/_b0_undertrained_600step`): layer-a used only
**5 / 256** codewords, 1 330 distinct SIDs, 571 colliding groups, loss 1.2142.
After the fix: **61 / 256** codewords, 3 675 SIDs, 11 colliding groups, loss 0.2725.
**The lesson is recorded in `configs/b0_b2_protocol.json`: take the budget from the launcher,
never from the parser.**

## 3. B2 — collaborative-alignment RQ-VAE

### 3.1 What LETTER actually does (read from source, not from the paper)

Upstream anchor: `github.com/HonghuiBao2000/LETTER`, commit `8d0154e2…` (12 files, per-file
sha256 in `_migration/letter_upstream/_upstream_manifest.json`).

| LETTER | File:line |
|---|---|
| contrastive loss | `RQ-VAE/models/rqvae.py:75-80` (`CF_loss`) |
| loss composition `+ alpha * cf` | `:107` |
| per-batch CF lookup by item index | `:103-104` |
| dataset returns `(emb, index)` | `datasets.py:15-18` |
| **CF table source** | `main.py:45,68` → **`<cat>-32d-sasrec.pt`** |

Two findings that change the implementation:

- ★ **The parameter named `quantized_rep` is not quantised.** At the call site (`:105`) it
  receives `dense_out`, the **encoder output before quantisation**. So LETTER's loss is an
  in-batch InfoNCE between the encoder's continuous output and a **frozen** per-item vector.
- ★ **LETTER's teacher is a trained SASRec**, not ItemCF/SVD.

### 3.2 Our implementation, and where it deliberately differs

```
L_total = L_rqvae + lambda_cf * L_cf          # L_rqvae = recon + quant_loss_weight * quant
L_cf    = InfoNCE( normalize(encoder(x)) , normalize(cf_table)[detached] , tau = 1.0 )
```

| Aspect | LETTER | **B2** | Why |
|---|---|---|---|
| teacher | SASRec item embedding | **ItemCF/SVD from TRAIN co-occurrence** | avoids training a second model; keeps the budget identical to B0 |
| similarity | raw dot product | **cosine ÷ temperature** | LETTER's unnormalised form makes the effective temperature magnitude-dependent; cosine + explicit τ keeps the loss scale interpretable and gradient-safe |
| no-evidence items | none | **masked out of the loss** | 97 items (2.6 %) have no reliable collaborative evidence; they keep the full reconstruction gradient |
| diversity loss | `vq.py:69-107` | **not implemented** | deliberately excluded so the ablation stays attributable |
| levels | 4 | **3** | matches MiniOneRec |

**Teacher construction (`build_collab_pairs.py`, TRAIN ONLY):**
per-row item **set** (dedup inside a window) → IDF weighting
`log((1+R)/(1+df))+1` → row L2 → `TruncatedSVD(32)` → item vectors L2-normalised.
Output: `cf_embedding_b2.npy` sha `3f0723530eb417e7`, shape (3686, 32), plus a boolean mask
`sid_b8697f067bc84bc0`. **3 589 items kept, 97 masked.**
`_assert_train_only()` raises if the path is not the train split, and the test suite asserts
that the module never names valid/test; Valid/Test are never opened.

★ **Stated approximation:** the train CSV is a *sliding-window* sample
(`st = max(i-10, 0)`), so we recover **window-level co-occurrence, not the user's full
interaction sequence**. Duplicates inside one row collapse to a set.

### 3.3 Why ItemCF/SVD instead of SASRec

1. **Budget parity.** A SASRec teacher needs its own training run and its own
   hyper-parameters; the comparison against B0 would then differ in two places at once.
2. **Determinism.** `TruncatedSVD(random_state=0)` on a fixed incidence matrix is exactly
   reproducible; a trained sequential model is not, without pinning another seed.
3. **Cost.** SVD over a 36 259 × 3 687 sparse matrix completes in seconds.
4. **Stated limitation.** It loses order information — a SASRec teacher encodes *sequence*,
   ours encodes *co-occurrence*. This is a design choice, not an equivalence.

### 3.4 Enforcing B0 equivalence when λ = 0

`train_rqvae.py` does not merely set the weight to zero: when `lambda_cf == 0` it constructs
the **unmodified upstream `Trainer`** over the **unmodified `EmbDataset`**. The collaborative
module is never imported, so it cannot perturb B0. Verified by unit test:

| check | result |
|---|---|
| α=0 → every one of 3 686 weights **exactly** 1.0 | ✅ |
| first-step loss, B0 vs λ=0 | 1.262883901596 vs 1.262883901596 |
| reconstruction loss | 1.058947563171 vs 1.058947563171 |
| **max gradient difference** | **0.000e+00** (bitwise) |
| parameter sets receiving gradients | 15 vs 15, identical |

### 3.5 Architecture and inference path

- **Encoder** `2560 → [2048,1024,512,256,128,64] → 32`; **3 residual quantisers**, 256
  codewords each, `e_dim = 32`; **decoder** mirrors the encoder. No trainable projection is
  introduced — the SVD dimension is fixed to `e_dim` and the trainer **asserts**
  `collab.dim == cfg.e_dim` rather than silently concatenating.
- **SFT:** Qwen2.5-0.5B, full-parameter, bf16, 2 epochs over
  `36 259 + 7 321 + 36 259 = 79 839` rows (next-item SID prediction + SID↔title + history→title).
- **Evaluation:** constrained beam search (`num_beams=20`, `max_new_tokens=256`,
  `length_penalty=0.0`) over a prefix trie built from **that arm's own** `info.txt`;
  SID depth is inferred from the catalogue (`sid_utils.infer_prefix_index`, → 3).

## 4. Results

### 4.1 RQ-VAE: multi-seed and λ ablation

| arm | λ_cf | seed | best_loss | first-pass collision | layer-a codewords | vocab |
|---|---|---|---|---|---|---|
| B0 | 0 | 2024 | 0.2725 | 0.1177 | 61 / 256 | 573 |
| B0 | 0 | 7 | 0.2175 | 0.1104 | 76 / 256 | 587 |
| B2 | 0.05 | 2024 | 0.6049 | 0.0925 | 141 / 256 | 653 |
| B2 | 0.10 | 2024 | 1.1027 | 0.1069 | 92 / 256 | 604 |
| **B2** | **0.20** | 2024 | 2.0120 | **0.0667** | **163 / 256** | 675 |
| B2 | 0.20 | 7 | 2.0330 | 0.0551 | 175 / 256 | 687 |

★ λ was **pre-registered by gradient-norm balance**, never tuned on a metric:
‖∇L_rqvae‖ = 0.0934, ‖∇L_cf‖ = 0.4727 ⇒ λ* = 0.1977 ⇒ used **0.2**; LETTER's default 0.1
sits in the same order of magnitude.

★ **Two-seed control:** layer-a usage spread **within B0 alone is 15 codewords**, versus the
B2 effect of ~100. That is why every comparison below is reported with **paired** bootstrap
rather than a seed-spread threshold.

★ **Collision-rate definition trap:** `best_collision_rate` is the trainer's **first-pass**
rate; the **shipped** catalogue is produced after up to 20 Sinkhorn retry rounds and
collapses to **11 groups / 22 items** for *every* arm. The shipped count therefore
**cannot distinguish arms** — verified A == B == D for all six arms.

### 4.2 S0 / S1 / S2 — formal recommendation experiment (TEST, 4 533 rows)

| arm | SID | λ_cf | **HR@20** | **NDCG@20** |
|---|---|---|---|---|
| *(P0, historical reference)* | *P0 index, config unknown* | — | *0.19832341* | *0.11786798* |
| **S0** | B0 seed 2024 | 0 | 0.19038165 | 0.11791880 |
| **S1** | B2 λ=0.10 | 0.10 | **0.19479373** | **0.12124519** |
| **S2** | B2 λ=0.20 | 0.20 | 0.18376351 | 0.11444928 |

★ **The italic P0 row is a frozen published reference, NOT a comparable arm** — P0's quantiser
checkpoint was never published, so its budget, seed and hyper-parameters are unknown
(`README.md:63`; artifact `checkpoints/grpo_evidence/eval_clean_sft/test_beam20.json`
sha `51903f9babf883fb`, which my re-implementation of `calc.py` reproduces to **Δ ≈ 4e-09**).
It is quoted only as a magnitude anchor: **B0/S0 is the only legitimate baseline in this
project.** S0 sits 0.79 pp below P0, and that gap is **not attributable** to anything.

| contrast | HR Δ (pp) | 95 % CI (row) | 95 % CI (user-clustered) | NDCG Δ (pp) | 95 % CI (row) | 95 % CI (user) |
|---|---|---|---|---|---|---|
| S1−S0 | +0.4412 | [−0.2647, +1.1692] ❌ | [−0.2697, +1.1430] ❌ | +0.3326 | [−0.0055, +0.6879] ❌ | [−0.0062, +0.6753] ❌ |
| S2−S0 | −0.6618 | [−1.3904, +0.0662] ❌ | [−1.3810, +0.0645] ❌ | −0.3470 | [−0.7056, +0.0064] ❌ | [−0.6839, −0.0135] ✓ |

★ **7 of 8 intervals contain zero.** The single exclusion is negative, and it is one of eight
comparisons. **Conclusion: no reliable evidence that B2 improves ranking.**

★ **Why S1's positive estimate is not credible anyway:** the paired transitions show
S1-wins **133** vs S1-loses **131**, and among the 729 rows both arms hit, S1 ranks better
**163** / worse **169** / equal **397**. The arms **re-shuffle the hit set**; they do not
find more items. A +0.44 pp point estimate from a ±0.7 pp interval is exactly what a
re-shuffle of that magnitude produces.

### 4.3 SID collisions and the evaluation convention

`calc.py` builds `sid → [item ids]` but uses it **only to count** unmapped predictions; the
score is pure **string equality** (`calc.py:66`). Consequences, measured on the real splits:

| quantity | VALID | TEST |
|---|---|---|
| colliding groups / items | 11 / 22 | 11 / 22 |
| rows whose **target SID is shared** | **207** | **261** |
| rows whose target is **not the lowest id** of its SID | **76** | **97** |
| **`identity_mismatch`** (SID hit resolving to a different item) | **0** | **0** |

The evaluator's implied item for a shared SID is the **lowest item id** — an **arbitrary but
fixed** policy. Because `identity_mismatch = 0`, the repo SID-level metric and the
Canonical-Item metric **coincide exactly** in this project, and no hit was ever credited to
the wrong item. That is a property of *this* catalogue, not a general guarantee.

**Metric implementation was verified, not assumed:** re-implementing `calc.py:34-81` on the
frozen P0 artifact reproduces the published baseline to **Δ ≈ 4e-09** with **20 beams** and
**0 unmapped predictions**. The NDCG gain `1/log(rank+2)` normalised by `1/log 2` is
**algebraically identical** to standard binary `1/log2(rank+1)`.

### 4.4 Phase 4A — long-tail diagnosis (VALID only)

| stratum | valid rows | S0 HR@20 | S1 HR@20 |
|---|---|---|---|
| freq0 (no train target) | 149 | 0.0201 | 0.0000 |
| low(1-2) | 448 | 0.0246 | 0.0335 |
| mid(3-20) | 2 719 | 0.1019 | 0.1048 |
| **high(>20)** | 1 216 | **0.4679** | 0.4622 |

**Exposure bias (S0):** Top-20 slots ÷ rows = freq0 **0.007×**, low **0.031×**, mid 0.535×,
high **2.519×** — a **360×** spread. The head (26.8 % of rows) takes **67.6 %** of the slots.

**Coverage vs ranking, separated:** global item coverage (S0) = **2 611/3 686 = 70.8 %**,
which is a **different quantity** from HR@20 (items vs requests) — subtracting them is
meaningless and that error was made and retracted. What *is* informative: in the mid stratum
**72.6 %** of rows have their target somewhere in *some* Top-20 yet only **10.2 %** hit, and
when the target *is* surfaced its median rank is **2**. So the failure is **entering the
candidate list**, not ordering within it.

**Mechanism verdict (each with support, counter-evidence and what stays undecided):**

| hypothesis | evidence | counter-evidence | status |
|---|---|---|---|
| **A. candidate exposure insufficient** | 0.007×–0.031× exposure ratios; 14 %/33 % coverage | mid is 72.6 % covered yet fails | **supported for the tail, not for mid** |
| **B. candidate ranking insufficient** | mid: 62 pp gap between coverage and HR | when surfaced, median rank is 2 | **supported for mid** |
| **C. training supervision insufficient** | next-item supervision 0 / 1.68 / 7.28 / 45.0 | all items have ≥1 *total* supervision; supervision and frequency are collinear | **plausible but not separable in this design** |
| **D. SID representation insufficient** | — | collision-group HR is a popularity artefact (colliding targets have median train freq 491 vs 8); no residual effect survives frequency control | **no supporting evidence under this design — NOT falsified** |

★ **Absence from a Top-20 is not evidence of zero probability.** Beam search returns 20
sequences; every "coverage" statement above is about the truncated candidate list only.

### 4.5 Phase 4B — frequency-aware loss (VALID only, TEST never run)

**What changed:** one scalar applied to the *next-item* supervision of `SidSFTDataset` and
`FusionSeqRecDataset`:

```
raw_weight(item) = (1 + train_next_item_frequency[item]) ** (-0.5)      # alpha = 0.5, pre-registered
weight(item)     = clip(raw_weight / mean(raw_weight), 0.25, 4.0)
```

`SidItemFeatDataset` keeps weight 1. Frequency is computed from **train only**.
Token aggregation: `loss_w = Σ_i w_i · Σ_t l_it / Σ_i w_i · n_i` where `n_i` is the number of
**supervised** tokens in sample i — so sequence length cannot change the meaning of the
weight. Prompt labels stay −100.

| **FINAL VALID RESULTS** | **C0 (α = 0)** | **C1 (α = 0.5)** | C1 − C0 |
|---|---|---|---|
| hits / rows | 760 / 4 532 | 736 / 4 532 | −24 |
| **HR@20** | **0.16769638** | **0.16240071** | **−0.5296 pp** |
| **NDCG@20** | **0.10862407** | **0.10422123** | **−0.4403 pp** |

| metric | unit | Δ (pp) | 95 % CI | excludes 0 |
|---|---|---|---|---|
| HR@20 | row | −0.5296 | [−1.2136, +0.1545] | ❌ |
| HR@20 | user (2 011) | −0.5296 | [−1.2815, +0.2435] | ❌ |
| NDCG@20 | row | −0.4403 | [−0.7412, −0.1213] | ✓ negative |
| NDCG@20 | user | −0.4403 | [−0.7829, −0.0799] | ✓ negative |

**Per stratum (user-clustered, all eight CIs exclude zero):**

| stratum | rows/users | C0 | C1 | HR Δ (pp) | HR 95 % CI | NDCG Δ (pp) |
|---|---|---|---|---|---|---|
| freq0 | 149/112 | 0.00000 | **0.03356** | **+3.3557** | [+0.6944, +6.4935] | +0.8352 |
| low(1-2) | 448/373 | 0.01116 | **0.02455** | **+1.3393** | [+0.4367, +2.4719] | +1.3851 |
| mid(3-20) | 2 719/1 508 | 0.07503 | **0.09526** | **+2.0228** | [+1.1029, +3.0737] | +0.7426 |
| **high(>20)** | 1 216/782 | 0.45312 | 0.37911 | **−7.4013** | [−9.2295, −5.6650] | −3.9141 |

**Exposure shift:** mid **+29.65 pp** (25.49 → 55.14), high **−30.31 pp** (74.39 → 44.08).
**Global item coverage: 1 898 (51.49 %) → 2 413 (65.46 %).**

★ **Interpretation.** The intervention **worked mechanically** — rare items got weight, the
candidate list broadened, and every tail stratum improved with a CI excluding zero. But the
head lost **7.40 pp**, which is larger in absolute terms, so the aggregate is
neutral-to-negative: overall HR unchanged (CI contains zero) and **overall NDCG
significantly worse**. This is a **head-vs-tail accuracy trade-off**, reported as such.
It is **not** evidence that exposure bias is causally inert — if anything the eight
consistent per-stratum intervals are evidence that exposure **is** causally manipulable.

### 4.6 ★ The bug that nearly produced a false negative (must be on the record)

The **first C1 run produced a model byte-identical to C0** (`5a6db35ceb74389d`) with the same
eval file and the same `train_loss`. Root cause: `transformers.Trainer` defaults to
`remove_unused_columns=True` and keeps only dataset columns present in `model.forward`'s
signature — the custom `_weight` column was **silently dropped**, so `compute_loss` never saw
it and the override fell through to the stock loss.

- Evidence: HF dataset columns `['input_ids','attention_mask','labels','_weight']` but
  `compute_loss` received `['attention_mask','input_ids','labels']`.
- Fix: `TrainingArguments.remove_unused_columns=False`. After the fix the weights arrive
  (`torch.float32 [0.5201, 0.3966, 1.0, 1.0]`) and weighted ≠ stock on **all 20 753**
  sampled batches (mean |Δ| 0.1836, max 2.8660).
- The void run is **quarantined**, not deleted: `/root/autodl-tmp/p4b_archive/void_C1/`.

★ **Had I not compared hashes, this would have been reported as "frequency reweighting has
no effect".** It is the single most transferable lesson in the project: **always verify that
an intervention actually reached the loss before interpreting a null.**

## 5. What was deliberately not done

| not done | reason | limitation it leaves |
|---|---|---|
| **B1** (text + collaborative vector **fusion**, then RQ-VAE) | the approved plan chose contrastive alignment; fusion alone has no mechanism to preserve text structure | we cannot say whether fusion would have beaten alignment |
| **B3/B4** (further λ or SID variants) | λ ablation already showed encoding-layer changes barely propagate to ranking | the λ→metric transfer function is mapped at only 3 points |
| **SASRec teacher** | budget parity and determinism (see §3.3) | order information is absent from the teacher; a sequence-aware teacher remains untested |
| **3-level vs 4-level SID** | 3 levels match MiniOneRec | LETTER's 4-level setting is untested here |
| **Diversity loss** | excluded to keep the ablation attributable | code-assignment bias is unmitigated |
| **TEST in Phase 4B** | pre-registered discipline: select on VALID only | 1-epoch conclusions rest on 4 532 valid rows |
| **GRPO / RL** | out of scope for this phase | no comparison against the RL line |

## 6. Final conclusions

1. **B2 measurably reshaped the SID codebook.** Layer-a codeword usage rose from 61/256 to
   163/256, first-pass collision fell from 0.1177 to 0.0667, and the alignment objective
   clearly learned (`pos_sim` −0.033 → +0.450). These are encoding-layer facts.
2. **That reshaping did not transfer into ranking accuracy.** S1's +0.44 pp HR@20 is inside
   its confidence interval; S2 trends negative. The transmission from SID structure to
   HR/NDCG is **weak**.
3. **The binding constraint is exposure, and it is manipulable.** Frequency-aware loss moved
   mid-band Top-20 exposure by +30 pp and improved **every** tail stratum with CIs excluding
   zero — while costing the head 7.4 pp and significantly worsening overall NDCG.
4. **Long-tail failure is two different problems.** For freq0/low items the target rarely
   enters the candidate list; for mid items it often *does* (72.6 % coverage) and then ranks
   poorly — but when it is surfaced its median rank is 2, so the loss happens at the
   boundary of the candidate set.
5. **No accuracy claim survives scrutiny, and that is the finding.** The project's value is
   a fully instrumented pipeline, a verified metric implementation, a documented
   collaboration-teacher recipe, and two well-measured negative/trade-off results.

## 7. Reproduction

**Read `REPRODUCE.md`.** The essential warning: **the local fork and the remote training repo
are different code**, so a single commit hash does not make every result reproducible.
`sft.py` / `data.py` / `evaluate.py` / `LogitProcessor.py` **differ** and only the **remote**
files produced the SFT results; `sid_utils.py` exists **only remotely**. The remote source is
now archived byte-exact and locally readable at
`deliverables/remote_source_archive/` (8 files, **8/8 hashes re-verified**), with the
container/GPU/dependency snapshot in `REMOTE_ENVIRONMENT.txt`.

**One archive gap, stated plainly:** the **five final SFT model weights**
(S0/S1/S2/C0/C1) currently exist **only on the AutoDL volume** — the local repo holds the
**evaluation artifacts**, not the weights. Mirroring the five `final_checkpoint` directories
(≈5.0 GiB) is the outstanding action. See `FINAL_ARCHIVE_VERIFICATION.md`.
