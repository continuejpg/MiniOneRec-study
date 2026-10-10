# LETTER SID — controlled comparison against a content-only baseline

Status: **frozen**. Single seed. No significance claims.

This document reports a controlled, system-level comparison between three
Semantic ID (SID) tokenizers feeding the *same* downstream pipeline, on the
Amazon `Industrial_and_Scientific` split.

---

## 1. Motivation

The repository's reference system (hereafter **P0**) uses a pre-computed 3-level
SID, `<a_x><b_y><c_z>`, produced upstream by constrained K-means
(`rq/rqkmeans_constrained.py`). Those artifacts are not reproducible from this
repository — no `.npy` / `.pth` / `.ckpt` for that tokenizer is present.

LETTER ([arXiv:2405.07314](https://arxiv.org/abs/2405.07314), CIKM'24) proposes a
*learnable* tokenizer combining

* a residual-quantized VAE for semantic regularization,
* an in-batch contrastive alignment to a collaborative embedding,
* a diversity loss to mitigate code-assignment bias.

Two questions follow, and they must be separated:

1. **Does a learnable 4-level tokenizer beat P0?**
2. **Does the full LETTER regularisation package add anything over the same
   encoder with the regularisers switched off?**

Question 2 needs a *content-only* control that differs from the LETTER treatment
**only** in `alpha` and `beta`. This repository reports that control (**B**).

---

## 2. Experimental setup

| Item | Value |
|---|---|
| Dataset | Amazon `Industrial_and_Scientific` |
| Catalogue | 3,686 items |
| Train / valid / test rows | 36,259 / 4,532 / 4,533 |
| Downstream model | Qwen2.5-0.5B, **full-parameter SFT** from the pristine base |
| Text encoder (frozen) | Qwen2.5-0.5B, masked-mean over `title` + `description` → `[3686, 896]` |
| Collaborative encoder (frozen) | SASRec-32 trained on train pairs → `[3686, 32]` |
| SID depth | 3 (P0) vs 4 (B, T) |
| Seed | 42, single seed |

The downstream recipe is identical for B and T (a strict copy of the P0 Clean SFT
recipe): 2 epochs, effective batch 64 (micro 16 × grad-accum 4), lr 3e-4, cutoff
512, bf16, `adamw_torch`, linear schedule, `warmup_steps=20`, no gradient
checkpointing. Evaluation is 4,533 test samples, beam 20, `num_return_sequences`
20, constrained decoding, `length_penalty=0`, `max_new_tokens=256`, batch 8, and
the same `calc.py`.

Only the SID representation, the SID index, the tokenizer and the output
directory differ between B and T. **No hyper-parameter was tuned for either
variant, and no epoch-count or learning-rate change was made in response to
intermediate results.**

---

## 3. Stage 1 — aligned text + CF embeddings

* `artifacts/letter_stage1/text_qwen05b.npy` — `[3686, 896]`, float32, 0 NaN/Inf,
  11 duplicate rows (explained: 11 groups share byte-identical cleaned text).
* `artifacts/letter_stage1/sasrec32_item_emb.pt` — `[3686, 32]`, float32, taken
  from `item_embeddings.weight[:3686]` of a `[3687, 32]` table (PAD row dropped).
* `artifacts/letter_stage1/cf_seen_mask.npy` — 3,647 train-seen / 39 train-unseen.

The **39 train-unseen items are excluded from tokenizer training batches** (they
carry no collaborative supervision, so including them would turn every
off-diagonal in-batch similarity into a false negative) but they **do** receive a
SID at inference time through the residual encoder and codebook.

Reproduce with `tools/letter_stage1/stage1_*.py`.

---

## 4. Stage 2 — LETTER tokenizer

Architecture and training configuration are identical for B and T except the two
regularisation coefficients:

| | **B (content-only)** | **T (LETTER treatment)** |
|---|---|---|
| `alpha` (collaborative) | **0** | 0.01 |
| `beta` (diversity) | **0** | 0.0001 |
| `num_emb_list` | `[256,256,256,256]` | `[256,256,256,256]` |
| `e_dim` | 32 | 32 |
| `layers` | `[2048,1024,512,256,128,64]` | same |
| `mu` (commitment) | 0.25 | same |
| `n_clusters` | 10 | same |
| `sk_epsilons` | `[0,0,0,0.003]` | same |
| lr / optim / weight-decay | 1e-3 / AdamW / 1e-4 | same |
| batch / seed / epochs | 1024 / 42 / **5000** | 1024 / 42 / **5000** |
| K-means init & per-epoch re-init | same | same |
| quantization / constrained K-means | same | same |
| checkpoint selection rule | same | same |

`rq/letter/` is the upstream implementation with five documented local edits
(M1–M5); the three loss definitions are byte-identical to upstream. See § Upstream
provenance.

### An intermediate result worth recording

An initial 200-epoch run produced a collapsed codebook (level utilisation
`7 / 22 / 76 / 156`, collision rate 0.552, and a collaborative objective
indistinguishable from chance: in-batch median rank 500/1024 against a chance
value of 512). Diagnosis showed this was a **training-length** effect, not a
defect in the loss: 200 epochs × 4 batches = **800** optimizer steps versus
200,000–400,000 for the upstream defaults. At **5,000 epochs** the same code
reaches `73 / 256 / 256 / 256` with collision rate 0.0033. Both B and T were then
run at 5,000 epochs.

---

## 5. Variable-depth SID engineering fixes

The 4-level path exposed four implicit `depth == 3` assumptions. All are fixed and
guarded by `tools/tests/test_sid_depth.py`.

| # | Location | Problem | Fix |
|---|---|---|---|
| 1 | `evaluate.py`, `LogitProcessor.py`, `minionerec_trainer.py` | `prefix_index` hardcoded (`4 if gpt2 else 3`) | derived from the catalogue via `sid_utils.infer_prefix_index` |
| 2 | `evaluate.py` | the evaluation tokenizer came from the checkpoint, which carries only that catalogue's SID tokens | register the catalogue's own SID tokens before building the trie |
| 3 | `data.py` `SidItemFeatDataset` | `sids[0]+sids[1]+sids[2]` silently dropped every level beyond the third | `"".join(sids)` — an identity for the 3-level catalogue |
| 4 | `LogitProcessor.py` | `ConstrainedLogitsProcessor` could not be told the depth | explicit `prefix_index` parameter |

`prefix_index` is **not** the SID depth: it is the token length of the prompt
prefix the entries carry. With the repository's `'### Response:\n'` wrapper that
is 3 tokens for both catalogues, which is why the constant happened to work for
P0 and silently constrained the 4-level trie to only its last transition.
`sid_utils` derives it from the catalogue and fails loudly on inconsistent depth,
empty SIDs, non-atomic levels, or a wrapper mismatch.

### P0 regression

`evaluate.py` on the same 2-row input, before and after these fixes:

```
sha256 before = 47bd83b90f5856cabc82f64b4d1edab3aa7923f89d450a5661d0c116b5a6e2d6
sha256 after  = 47bd83b90f5856cabc82f64b4d1edab3aa7923f89d450a5661d0c116b5a6e2d6
BYTE-IDENTICAL = True
```

---

## 6. Content-only control (B)

B is produced by **configuration only** — `--alpha 0.0 --beta 0.0` on the same
training entry point. No LETTER code was removed or rewritten.

Because both coefficients multiply their term, setting them to zero zeroes the
contribution rather than deleting the term. That was verified by **perturbation**
rather than by re-deriving the values (the diversity term is stochastic — it
samples a positive target with `random.choice` each forward):

| Perturbation | `alpha=0` | control `alpha=100` |
|---|---|---|
| replace `cf_embedding` with an independent random matrix (relative change 1.41) | `n_diff = 0`, `max = 0.0` | 14 params differ, `max = 1.05e+02` |

| Perturbation | `beta=0` | control `beta=100` |
|---|---|---|
| permute 826/1024 code→cluster assignments | `n_diff = 0`, `max = 0.0` | 4 params differ, `max = 5.54e-01` |

Every parameter gradient is **bitwise identical** under both perturbations at the
B setting, and the controls confirm the test is not vacuous. Reproduce with
`tools/letter_stage1/stage4_step2_grad_proof.py`.

### Tokenizer-level outcome

| | **B** | **T** |
|---|---|---|
| `alpha` | 0 | 0.01 |
| `beta` | 0 | 0.0001 |
| depth | 4 | 4 |
| codebook | 4 × 256 | 4 × 256 |
| text encoder | same | same |
| seed | 42 | 42 |
| epochs | 5,000 | 5,000 |
| runtime | 5,259.4 s | 5,276.4 s |
| peak VRAM | 0.205 GiB | 0.205 GiB |
| final reconstruction loss | **0.358129** | 0.416909 |
| **unique SID** | **3,675** | **3,674** |
| **collision rate** (final) | **0.002984** | **0.003256** |
| **level utilisation** | **34 / 256 / 256 / 256** | **73 / 256 / 256 / 256** |
| collision groups / affected items | 11 / 22 | 11 / 23 |
| max collision size | 2 | 3 |
| distinct SID tokens | 802 | 841 |
| final tokenizer length | 152,467 | 152,506 |

Both catalogues cover 3,686/3,686 items with exactly 4 tokens each and no missing
mapping, NaN or Inf. B concentrates level `a` more strongly (13.3 % utilisation
vs T's 28.5 %) while levels `b`–`d` are fully used in both.

The monitored `diversity` and collaborative loss values for B are recorded in
`artifacts/letter_stage4_content_only/train_history.json`, but because their
coefficients are zero **they are not part of B's optimisation objective**.

---

## 7. Downstream results

Both variants were trained from the pristine Qwen2.5-0.5B base and evaluated with
their own tokenizer, info file, test file and trie. LegalRate and DuplicateRate
were 100.0000 % and 0.0000 % for both (90,660 candidates each).

### Full HR / NDCG table

| metric | **P0** | **B (content-only)** | **T (LETTER)** |
|---|---|---|---|
| **HR@1** | 0.06926980 | 0.07081403 | **0.07434370** |
| **HR@3** | **0.10103684** | 0.09508052 | 0.09905140 |
| **HR@5** | **0.12111185** | 0.11228767 | 0.11780278 |
| **HR@10** | **0.15376131** | 0.13765718 | 0.14537834 |
| **HR@20** | **0.19832341** | 0.16765939 | 0.17383631 |
| **NDCG@1** | 0.06926980 | 0.07081403 | **0.07434370** |
| **NDCG@3** | 0.08789727 | 0.08494025 | **0.08840169** |
| **NDCG@5** | **0.09613706** | 0.09202226 | 0.09609076 |
| **NDCG@10** | **0.10663781** | 0.10023594 | 0.10495113 |
| **NDCG@20** | **0.11786798** | 0.10784649 | 0.11208197 |

### T relative to B

| metric | absolute pp | relative % |
|---|---|---|
| HR@1 | +0.3530 | +4.9844 |
| HR@3 | +0.3971 | +4.1763 |
| HR@5 | +0.5515 | +4.9116 |
| HR@10 | +0.7721 | +5.6090 |
| **HR@20** | **+0.6177** | **+3.6842** |
| NDCG@1 | +0.3530 | +4.9844 |
| NDCG@3 | +0.3461 | +4.0751 |
| NDCG@5 | +0.4069 | +4.4212 |
| NDCG@10 | +0.4715 | +4.7041 |
| **NDCG@20** | **+0.4235** | **+3.9273** |

### B relative to P0

| metric | absolute pp | relative % |
|---|---|---|
| HR@1 | +0.1544 | +2.2293 |
| **HR@20** | **−3.0664** | **−15.4616** |
| NDCG@1 | +0.1544 | +2.2293 |
| **NDCG@20** | **−1.0021** | **−8.5023** |

**T relative to B improves all reported HR/NDCG cutoffs, but neither B nor T beats
P0 on HR@20/NDCG@20.** The ranking is P0 > T > B on both headline metrics.

---

## 8. Interpretation

Under this configuration, in a single-seed controlled comparison:

* The content-only 4-level tokenizer (B) does **not** exceed the existing P0 SID
  system; it is lower on HR@20 (−3.07 pp) and NDCG@20 (−1.00 pp), and higher only
  at rank 1.
* The full LETTER regularisation package (T) is **better than B at every reported
  cutoff** (HR@20 +0.62 pp, NDCG@20 +0.42 pp), but still **below P0** on both
  headline metrics (HR@20 −2.45 pp, NDCG@20 −0.58 pp).

Safe statement, used verbatim in reporting:

> Under the same text encoder, 4-level codebook, training pipeline and seed, the
> full LETTER regularisation package performs better than the content-only
> baseline, but does not exceed the existing P0 SID system.

**Not claimable from this experiment.** T versus B changes `alpha` *and* `beta`
together, so the comparison attributes to the *complete LETTER regularisation
package*; it does not isolate the collaborative term, does not isolate the
diversity term, and does not establish that any collaborative signal alone is
responsible for the observed difference. No significance statement is made: this
is a single-seed, system-level comparison.

---

## 9. Limitations

* **L1 — single seed.** No variance estimate, no significance testing.
* **L2 — one dataset / one category.** `Industrial_and_Scientific` only.
* **L3 — one downstream scale.** Qwen2.5-0.5B only. A larger backbone was not
  part of this comparison.
* **L4 — environment.** Installing `k-means-constrained` for the LETTER tokenizer
  upgraded `protobuf` to 6.33.6, which `tensorboard 2.11.2` cannot import
  (`TypeError: Descriptors cannot be created directly`). Runs use protobuf's
  documented pure-Python workaround
  (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) together with an inert `wandb`
  stub. This affects logging only, not the reported metrics, but it means the
  environment is not pristine.
* **L5 — `SidItemFeatDataset` size varies by catalogue** (7,316 P0 / 7,320 T /
  7,321 B) because `title2sid` is a dict keyed by title and identical titles
  collapse. No sample was added or removed to force the counts to match.
* **L6 — B's level-`a` utilisation (34/256) is much lower than T's (73/256).**
  This is reported as an observation; it was not treated as a reason to change
  the training recipe.
* **L7 — P0 remains non-reproducible in this repository**; its SID artifacts
  originate upstream and no tokenizer checkpoint is present, so P0 is a fixed
  reference point rather than a matched control.

---

## 10. Reproduction

```bash
# 0. dependency: reconstruct the LETTER tokenizer sources (no license to vendor)
python tools/fetch_letter_upstream.py            # -> rq/letter/, hash-verified

# 1. Stage 1 — frozen embeddings
python tools/letter_stage1/stage1_build_item_manifest.py
python tools/letter_stage1/stage1_text_embedding.py
python tools/letter_stage1/stage1_sasrec32_embedding.py

# 2. Stage 2 — treatment tokenizer T (5000 epochs)
python tools/letter_stage1/stage2_train_letter.py --epochs 5000 \
    --out-dir artifacts/letter_stage2
python tools/letter_stage1/stage2_generate_sid.py \
    --ckpt artifacts/letter_stage2/letter_tokenizer_best_collision.pth \
    --out-dir artifacts/letter_stage2

# 3. Stage 4 — content-only control B (identical except the two coefficients)
python tools/letter_stage1/stage2_train_letter.py --epochs 5000 \
    --alpha 0.0 --beta 0.0 --out-dir artifacts/letter_stage4_content_only
python tools/letter_stage1/stage4_generate_sid.py

# 4. remap, tokenizer, preflight  (repeat per variant)
python tools/letter_stage1/stage3_step34_remap.py \
    --letter-index artifacts/letter_stage2/letter_index.json \
    --out-root     artifacts/letter_stage3/data
python tools/letter_stage1/stage3_step1_tokenizer.py \
    /path/to/Qwen2.5-0.5B \
    artifacts/letter_stage2/letter_index.json \
    artifacts/letter_stage3
PREFLIGHT_STAGE=3 python tools/letter_stage1/stage3_step5_preflight.py

# 5. SFT + formal eval
bash sft_letter_full.sh          #  or sft_content_only_full.sh
bash eval_letter.sh              #  or eval_content_only.sh
python tools/letter_stage1/stage3_step8_legality.py     # legality / duplicates

# 6. comparison table
python tools/letter_stage1/stage4_step13_compare.py

# 7. regression (CPU only, seconds, loads no model)
python tools/tests/test_sid_depth.py
```

---

## Upstream provenance and licensing

* **Repository**: <https://github.com/HonghuiBao2000/LETTER>
* **Commit**: `8d0154e28de37dbb6e24871c508ad8ddb1921cda` (branch `master`)
* **Files used**: `RQ-VAE/{main,trainer,datasets,utils,generate_indices}.py`,
  `RQ-VAE/models/{rqvae,rq,vq,layers}.py`

**The upstream project declares no license.** Its tree contains no `LICENSE`,
`COPYING`, `NOTICE` or `COPYRIGHT` file; GitHub reports `license: None`; and the
README contains only a citation request. Because no redistribution permission is
granted, the upstream source is **not committed to this repository**.

In its place:

* `tools/fetch_letter_upstream.py` reconstructs `rq/letter/` from the pinned
  commit via the GitHub API, verifies the SHA256 of every verbatim file against
  the values recorded when this work was done, applies the five local edits
  (M1–M5) with exact-match anchors, and writes `UPSTREAM_MANIFEST.json`.
  Verified to reproduce the runtime tree byte-for-byte (9/9 files).
* The three loss definitions used in the experiments are **unchanged** from
  upstream: `CF_loss`, `diversity_loss`, and
  `loss = codebook_loss + mu * commitment_loss + beta * diversity_loss` /
  `total_loss = rqvae_loss + alpha * cf_loss`.

If the upstream authors later publish a license permitting redistribution, the
vendored directory can be committed as-is; the manifest and patch list above are
sufficient to verify it.
