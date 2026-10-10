# GRPO Post-training Results — P0 / optimized GRPO / R1 / R2

Experiment log for every RL run in this repository, in chronological order.

> **Bottom line: none of the RL methods beat the SFT baseline.**
> This is a negative result, reported in full. Mechanism explanations below are
> **hypotheses consistent with the data — not proven causes**. All runs are
> **single-seed**, so no significance claim is made anywhere in this document.

## Split protocol — read this first

| split | samples | where used |
|---|---|---|
| **test** | 4,533 | historically the *only* evaluated split (all runs before R2) |
| **validation** | 4,532 | introduced during R2.2 for a same-protocol 3-way comparison |

**The two splits are not interchangeable and their numbers must never be mixed.**
Every table below states its split explicitly. The validation baselines for
SFT and old-GRPO were produced during R2.2 by running the *same* evaluation
script against the same split, so §4's three-way comparison is internally
consistent; it is **not** comparable with the test-split tables.

Shared evaluation protocol: `evaluate.py` beam search, `num_beams=20`,
`max_new_tokens=256`, `length_penalty=0.0`, `batch_size=8`, metrics by `calc.py`.

---

## 1. P0 SFT — the best model

**Goal.** Establish the reference: content-only Semantic ID, Qwen2.5-0.5B SFT,
constrained decoding.

**Change.** None relative to upstream. SID produced by the content-only RQ
pipeline (48/256/256 level cardinality), frozen; `TokenExtender` adds the SID
vocabulary so each `<x_N>` is one atomic token.

**Training.** `bash sft_full.sh` — Qwen2.5-0.5B, single RTX 4090.

**Evaluation protocol.** test split (4,533), beam 20.

**Data.**

| metric | HR@1 | HR@3 | HR@5 | HR@10 | **HR@20** |
|---|---|---|---|---|---|
| P0 SFT | 0.06926980 | 0.10103684 | 0.12111185 | 0.15376131 | **0.19832341** |

| metric | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | **NDCG@20** |
|---|---|---|---|---|---|
| P0 SFT | 0.06926980 | 0.08789727 | 0.09613706 | 0.10663781 | **0.11786798** |

**Conclusion.** SFT alone is the strongest configuration in this repository.

---

## 2. Optimized GRPO — efficiency only, no quality gain

**Goal.** Make GRPO cheap enough to iterate on, without changing what it learns.

**Change.** Batch / gradient-accumulation restructuring only
(`patches/grpo_fast025.sh` vs the earlier 0.25-epoch launcher). Reward, β, LR, G,
data and epoch count are identical.

**Training configuration (both runs identical except batch structure).**

```
reward_type             ranking   (rule_reward + ndcg_rule_reward)
num_generations         16
beam_search             True            -> BEAM_SAMPLE
train_batch_size        16
gradient_accumulation_steps 1
num_train_epochs        0.25
learning_rate           1e-5
beta                    1e-3
temperature             1.0
sync_ref_model          True
add_gt / dapo / gspo    False
data                    17,516 frozen samples
                        seq_rec 10,000 + title2sid/description2sid 6,516 + seqtitle2sid 1,000
seed                    42
steps                   4,379
```

**Evaluation protocol.** test split (4,533), beam 20.

**Data — throughput.**

| run | runtime | steps | s/step | steps/s |
|---|---|---|---|---|
| pre-optimization 0.25 ep | 6,111.6 s | 4,379 | 1.396 | 0.717 |
| **optimized 0.25 ep** | **1,598.2 s** | 4,379 | **0.365** | **2.740** |

**3.82× speed-up** (6,111.6 / 1,598.2 = 3.824). Peak VRAM 7.63 GB.

**Data — quality.**

| metric | **HR@20** | **NDCG@20** |
|---|---|---|
| optimized GRPO 0.25 ep | 0.17538054 | 0.10824989 |

> **Provenance.** These values come from
> [`notes/experiment_summary.md`](../notes/experiment_summary.md), the repository's
> authoritative machine-derived record, which lists the full grid as
>
> ```
> | GRPO 0.25 ep (optimized) | grpo_fast025/checkpoint-4379 | 0.06750496 |
>   0.09596294 | 0.11030223 | 0.14096625 | 0.17538054 | 0.10824989 |
> ```
>
> No `metrics.txt` is stored for this run - the run directory keeps
> `eval_beam20.json` (the raw prediction artifact) plus `train.log`. The record
> states every headline number was re-derived from its artifact rather than copied,
> and `analysis/verify_summary_claims.py` re-checks it.

**Comparison (conditional on the unverified numbers above).** vs SFT: **HR@20 −2.2943 pp / −11.57%**, NDCG@20 −0.9618 pp / −8.16%.

**Failure hypothesis.** The `ranking` reward is a **dense rank reward**: only
groups containing the ground truth receive a positive signal (7.76% of groups in
the R1 reproduction), while the remaining groups get an all-zero reward vector.
Measured `zero_advantage_group_ratio` for this reward family is **0.7015**
(bimodal: `{0.0: 1307, 1.0: 3072}`) — i.e. ~70% of optimizer steps carried no
group-relative signal.

**Conclusion.** Optimized GRPO is a **valid, useful efficiency baseline** (3.82×
faster, identical configuration), but it **does not improve ranking quality** over
SFT on this data.

---

## 3. R1 — ReRe-style rank reward

**Goal.** Test whether a *rank-aware* reward (ReRe Eq.7–9) fixes the
zero-advantage problem while preserving ranking quality.

**Change.** New pure-function reward module `rere_reward.py`; a
`rere_rank` reward type registered in `rl.py`. Upstream `ranking` reward left
untouched.

```
Eq.7  R_rule_i      = +1 if candidate i is the GT else 0
Eq.8  R_hat_rank_i  = 0 if GT, else -1/log2(r+2)          (r = 0-based rank)
Eq.9  R_rank_i      = -R_hat_rank_i / sum_j R_hat_rank_j   (per group)
      R_i           = R_rule_i + R_rank_i
```

**Training configuration.** Strict copy of the optimized launcher; only
`--reward_type rere_rank` and `--output_dir` differ (verified by `diff`).

**Evaluation protocol.** test split (4,533), beam 20.

**Data — training.**

| metric | value |
|---|---|
| runtime | 1,560.44 s (2.806 step/s, 4,379 steps) |
| `zero_advantage_group_ratio` | **0.0** — every step (4,379/4,379) |
| `group_reward_std_mean` | mean 0.0503 / median 0.031992 |
| GT-hit group ratio (derived: `sum(R_total) = n_hits − 1`) | **340/4,379 = 7.76%** |
| KL | median 0.0113, p99 3.136, max 7.73e5 |

**Data — quality.**

| metric | P0 SFT | **R1** |
|---|---|---|
| HR@1 | 0.06926980 | 0.04875358 |
| HR@3 | 0.10103684 | 0.04963600 |
| HR@5 | 0.12111185 | 0.05095963 |
| HR@10 | 0.15376131 | 0.05338628 |
| **HR@20** | **0.19832341** | **0.05912199** |
| **NDCG@20** | **0.11786798** | **0.05203455** |
| LegalRate | 100% | 100% |
| DuplicateRate | 0% | 0% |

**R1 vs SFT: HR@20 −13.9201 pp / −70.19%, NDCG@20 −6.5833 pp / −55.85%.**

**Mode collapse (test split).**

| model | distinct top-1 predictions | most-common top-1 |
|---|---|---|
| P0 SFT | 1,059 | — |
| optimized GRPO | 1,252 | — |
| **R1** | **99** | **1,522 / 4,533 = 33.6%** |

**Failure hypothesis (data-consistent, NOT proven causal).** Eq.9 normalises the
rank term *within each group*, so a candidate is penalised for being out-ranked
by its own group siblings rather than for being wrong. In the 92.2% of groups
with no ground truth this produces a reward that is maximised by *keeping the
current ordering*, i.e. self-reinforcement of the frozen policy's own ranking.
The observed collapse to 99 distinct predictions is consistent with that
mechanism. Because there is no control run (e.g. a rank reward with a GT-only
term), **causality is not established**.

**Conclusion.** Removing zero-advantage groups entirely (0.0) did **not** help;
quality collapsed. **R1 is a negative result.**

---

## 4. R2 — reachability-guided GRPO

**Goal.** Test whether the ground truth is *reachable at all* under the
constrained decoder, and whether giving the model a partial hint recovers
otherwise-unreachable samples.

### 4.1 Design

- **NORMAL (h=0)** — prompt unchanged; completion compared to the GT SID.
- **HARD (h=1)** — the **first GT SID token** is appended to the prompt, after the
  `### Response:` marker (the answer slot). The completion therefore holds only the
  sampled suffix; the full SID is reconstructed as `hint + suffix` before the
  exact-match comparison.
- **Reward** — **exact-match 0/1 only.** The R1 rank reward is not used.
- **Prefix CE** — for HARD groups the hinted token would otherwise never be
  trained, so a supervised cross-entropy on the hint token is computed from the
  **original, un-hinted prompt** and added with a **fixed coefficient 0.1**.
  Hinted tokens are inside `prompt_ids`, so they receive **no GRPO/KL credit**.
- **Route cache** — computed offline, once, from the frozen SFT model's h=0
  rollout on the **train split only**, keyed by stable `sample_id`
  (`splits/r21_route_cache.json`, 17,516 entries).

### 4.2 Why the `count_0` fix was required

`ConstrainedLogitsProcessor` hardcoded `self.count = 0`, so its first trie key was
always `sent[-3:]`. With a hinted prompt that window slides onto the hint:

```
h=0  key = ['###', 'ĠResponse', ':Ċ']            -> HIT,  48 candidates
h=1  key = ['ĠResponse', ':Ċ', '<a_236>']        -> MISS -> allowed=[] -> forced EOS
h=2  key = [':Ċ', '<a_236>', '<b_231>']          -> MISS -> allowed=[] -> forced EOS
```

The trie itself was fine (`hash(SID[:1])` and `hash(SID[:2])` both exist and
return legal continuations). Adding `count_0` — the number of SID tokens the
prompt already ends with — and counting the hinted tokens as already generated
fixes it. **`count_0` defaults to 0, so the NORMAL path is bit-for-bit unchanged**
(verified: identical masks at G=1 and G=16).

### 4.3 Route cache

| route | count | share |
|---|---|---|
| NORMAL (h=0) | 4,818 | 27.51% |
| HARD (h=1) | 12,698 | 72.49% |
| **total** | **17,516** | 100% |

| task | NORMAL | HARD | total | HARD% |
|---|---|---|---|---|
| `seq_rec` | 2,595 | 7,405 | 10,000 | 74.05% |
| `title2sid` | 1,914 | 1,732 | 3,646 | 47.50% |
| `description2sid` | 182 | 2,688 | 2,870 | 93.66% |
| `seqtitle2sid` | 127 | 873 | 1,000 | 87.30% |

### 4.4 Training configuration

Strict copy of `patches/grpo_fast025.sh`; only `--reward_type r21_exact`,
`--r21_enable True`, `--route_cache`, `--output_dir` differ.
SFT checkpoint, seed, G, beam_sample, KL β, LR, batch, grad-accum, subsets and
0.25 epoch are unchanged.

**Evaluation protocol.** **validation** split (4,532), beam 20.

### 4.5 Data — training

| metric | value |
|---|---|
| runtime | 3,279.07 s (1.335 step/s, 4,379 steps) |
| peak VRAM | 16.74 GB (reserved 22.41 GB) |
| train_loss | 0.783295 |
| NaN / Inf | **0** (all 4,379 steps finite) |
| grad_norm | median 13.24, p99 86.58, max 1,158,557 |
| KL | median 0.0658, p99 105.88, max 1,157,928 |
| `zero_advantage_group_ratio` | mean **0.4750**, median 0.0 |
| GT-hit candidate fraction | **3.2813%** (= 2,299 / 70,064) |

Per-task zero-advantage ratio: `seq_rec` 0.4417, `title2sid` 0.3827,
`description2sid` 0.6596, `seqtitle2sid` 0.6109.

### 4.6 Data — validation, three-way, same protocol

| model | HR@1 | HR@3 | HR@5 | HR@10 | **HR@20** | NDCG@10 | **NDCG@20** |
|---|---|---|---|---|---|---|---|
| **P0 SFT** | 0.07965578 | 0.11010591 | 0.12577229 | 0.15379523 | **0.18314210** | 0.11271205 | **0.12011306** |
| optimized GRPO | 0.07722860 | 0.10458959 | 0.12091792 | 0.14474846 | **0.17122683** | 0.10779942 | **0.11443492** |
| **R2** | 0.07568402 | 0.09708738 | 0.10834069 | 0.12996470 | **0.15600177** | 0.09977075 | **0.10630817** |

**Ranking: SFT > optimized GRPO > R2.**

| comparison | HR@20 | NDCG@20 |
|---|---|---|
| R2 vs SFT | **−2.7140 pp / −14.82%** | −1.3805 pp / −11.49% |
| R2 vs optimized GRPO | **−1.5225 pp / −8.89%** | −0.8127 pp / −7.10% |

R2 is behind on **every** K ∈ {1,3,5,10,20}, i.e. a consistent deficit rather than
a single noisy point.

**Mode-collapse check (validation).**

| model | distinct top-1 | most-common top-1 | distinct candidates (of 90,640) | LegalTop1 |
|---|---|---|---|---|
| P0 SFT | 1,134 | 498 (10.99%) | 2,571 | 100% |
| optimized GRPO | 1,339 | 664 (14.65%) | 3,296 | 100% |
| **R2** | 1,223 | **952 (21.01%)** | 3,047 | 100% |

No severe collapse (R2 still emits 1,223 distinct top-1 and 3,047 distinct
candidates, 100% legal), but top-1 concentration rose to **1.91× the SFT level**.
For reference, R1's collapse was far worse (99 distinct top-1, 33.6% share).

**Failure hypothesis (data-consistent, NOT proven causal).** Zero-advantage
groups improved substantially (0.7015 → 0.4750), yet quality fell. That is
evidence that **zero-advantage ratio was not the bottleneck** — old GRPO scored
better while carrying ~70% zero-advantage groups. The hint also hands the model
the first SID level for free on 72.49% of samples, which plausibly shifts learning
toward the residual suffix only. Two confounded changes were applied at once
(exact-match reward *and* hint routing), and there is **no ablation separating
them** — so the cause of the regression is **UNKNOWN**.

**Conclusion.** Reachability is real (oracle reachability is 100% by construction;
the h=1 hint recovers a large share of h=0 misses in the R2.0 probe), and the
implementation is verified correct — but the configuration **did not improve
ranking quality**. **R2 is a negative result.** Test-split evaluation was
deliberately **not** run: the validation deficit is consistent across all K, so
further test compute was not justified.

---

## 5. What the RL experiments jointly show

| claim | status |
|---|---|
| optimized GRPO is 3.82× faster than the pre-optimization 0.25-epoch run | **measured** |
| optimized GRPO does not beat SFT | **measured** (test) |
| R1 collapses the output distribution | **measured** (test) |
| R2 lowers zero-advantage ratio from 0.7015 to 0.4750 | **measured** (train) |
| R2 does not beat SFT or old GRPO | **measured** (validation) |
| lowering zero-advantage group ratio is *not* sufficient for quality | **supported by the data** |
| the ReRe Eq.9 within-group normalisation *causes* R1's collapse | **HYPOTHESIS — no control run** |
| hint routing *causes* R2's regression | **UNKNOWN — confounded with the reward change** |
| any of these differences is statistically significant | **NOT TESTED — single seed per configuration** |

### 5.1 Highest-value next experiment (not run)

Ablate the two R2 changes independently:

- `--r21_enable True` with a cache whose every entry is **NORMAL** → pure
  exact-match GRPO, no hint.
- the same cache with every entry **HARD**.

This separates "exact-match reward replaced the ranking reward" from "hint
routing" — the single largest information gap in this log.

---

## 6. Artifact locations

| artifact | path |
|---|---|
| P0 SFT checkpoint (**best**) | `runs/industrial_sft/final_checkpoint` |
| optimized GRPO checkpoint | `/root/autodl-tmp/runs/grpo_fast025/checkpoint-4379` |
| R1 checkpoint | `runs/grpo_rere_rank025/final_checkpoint` |
| R2 checkpoint | `runs/grpo_r21/final_checkpoint` |
| R2 route cache | `splits/r21_route_cache.json` |
| test-split metrics (SFT) | `runs/reference_full_clean_beam20/metrics.txt` |
| test-split metrics (R1) | `runs/eval_grpo_rere_rank025/metrics.txt` |
| validation metrics (SFT / old GRPO / R2) | see `docs/reproducibility.md` |

Checksums (SHA256) for all of the above are in
[`docs/reproducibility.md`](reproducibility.md) §6.
