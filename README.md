<div align="center">

<img src="./assets/logo.png" width="420em" ></img>

**MiniOneRec — generative recommendation reproduction, Semantic-ID intervention analysis, and single-GPU training/inference optimization. All on one RTX 4090.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
![GPU](https://img.shields.io/badge/GPU-1%C3%97%20RTX%204090-orange.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Upstream%20Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 Upstream Technical Report</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Upstream Huggingface</a> | <a href="docs/notes/experiment_summary.md">📊 Full Experiment Record</a>

</div>

> **Relationship to upstream.** This is an independent reproduction and analysis of
> [MiniOneRec](https://github.com/AkaliKong/MiniOneRec), not the upstream project.
> All upstream attribution, citation and license information is preserved at the bottom of
> this file — see [Upstream framework & attribution](#upstream-framework--attribution).
>
> **What this is.** A research *record*, not a one-command reproduction. Every headline
> number is traced to a prediction artifact and graded by how far that trace reaches; the
> limits are stated next to the results, not buried. Negative results are kept in full.
>
> **★ This is a curated release snapshot, and it ships without data.** The entire
> Amazon-derived corpus (split CSVs, SID index, item metadata, info files) is **excluded**
> because its redistribution terms are **not confirmed**. **Read [`DATA.md`](DATA.md)** for
> what was removed, how to obtain it, and exactly which parts of this repository do and do
> not work without it.

---

## 1. Project Overview

MiniOneRec turns recommendation into **constrained sequence generation**: every item is
compressed into a 3-token **Semantic ID** (SID), an LLM is fine-tuned to emit the next
item's SID, and a prefix trie guarantees every beam decodes to a legal item.

The open question this repository attacks is a **causal** one:

> **Does the model learn SID's semantic structure, or does it just memorise the
> item ↔ SID table?** SIDs are an *arbitrary* assignment, so a high score alone cannot
> distinguish the two.

### Three core contributions

| | Contribution | Headline |
|---|---|---|
| **① Diagnose** | A **strict controlled intervention** on the item↔SID correspondence that holds the codebook, vocabulary, trie, collision structure, popularity strata and recipe fixed. | **HR@20 −45.72 %** |
| **② Reproduce & fix** | Reproduced SFT → GRPO → constrained decoding end to end, and fixed correctness bugs (notably reward↔sample binding) that made the RL results unattributable. | RL results made interpretable |
| **③ Optimize** | Cut GRPO wall-clock by removing repeated in-training validation and redundant saves — **final evaluation protocol untouched**. | **wall-clock −73.8 %, throughput 3.82×** |

**What this is not:** not a SOTA claim, not a paper summary, and not a full replication.
It is a single-seed, single-category *diagnostic study* whose value is the control it
exercises and the honesty of its limits.

> **Where do I start an experiment?** **[`docs/ENTRYPOINTS.md`](docs/ENTRYPOINTS.md)** maps
> each of the five study groups to its real launcher, call chain, configuration, artefacts
> and known gaps.
>
> | # | study group | canonical entry point |
> |---|---|---|
> | ① | Clean SFT baseline | [`sft_full.sh`](sft_full.sh) |
> | ② | shuffled-SID intervention | [`scripts/sft_shuffled_sid.sh`](scripts/sft_shuffled_sid.sh) → [`scripts/eval_shuffled_sid.sh`](scripts/eval_shuffled_sid.sh) |
> | ③ | sequence-only ablation | [`scripts/sft_seq_only_clean.sh`](scripts/sft_seq_only_clean.sh) (`--sft_mode seq_only`) |
> | ④ | Qwen2.5-1.5B scaling check | **no launcher exists** — see [`docs/ENTRYPOINTS.md`](docs/ENTRYPOINTS.md) §4 |
> | ⑤ | GRPO / R1 / R2 (negative results) | [`patches/grpo_baseline.sh`](patches/grpo_baseline.sh) … [`patches/grpo_r21.sh`](patches/grpo_r21.sh) |

---

## 2. Key Findings

Formal protocol: deterministic constrained beam search, `num_beams` 20, `max_new_tokens`
256, `length_penalty` 0, metrics by the unmodified `calc.py`, `K = [1, 3, 5, 10, 20]`.

**Split discipline — `test` (4 533 samples) and `validation` (4 532) are not
interchangeable and are never mixed below.** Every row except R2 is on test; R2 and the two
baselines produced for it are on validation.

| # | System | Split | HR@20 | NDCG@20 | Grade |
|---|---|---|---|---|---|
| 1 | Compact SASRec (Protocol B) | test | 0.07147584 | 0.05156672 | E3 |
| 2 | **Clean SFT** | test | **0.19832341** | **0.11786798** | E3 |
| 3 | GRPO 0.25 ep, original | test | 0.17626296 | 0.10885475 | ★ **E4** |
| 4 | GRPO 0.25 ep, optimized | test | 0.17538054 | 0.10824989 | E1′ |
| 5 | GRPO 1.5 ep | test | 0.16170307 | 0.10470676 | E3 |
| 6 | GRPO 2.0 ep | test | 0.16236488 | 0.10447064 | E3 |
| 7 | **Shuffled-SID SFT** | **shuffled** | **0.10765497** | 0.08157358 | E2 |
| 8 | Seq-only clean | test | 0.17802780 | 0.10842814 | E1 |
| 9 | Seq-only shuffled | **shuffled** | 0.10125745 | 0.07683363 | E1 |
| 10 | 1.5B clean | test | 0.19964703 | 0.11877667 | E1 |
| 11 | 1.5B shuffled | **shuffled** | 0.11625855 | 0.08510443 | E1 |
| 12 | **R1** (ReRe rank reward) | test | **0.05912199** | 0.05203455 | E1′ |
| 13 | **R2** (reachability-guided) | ★ **validation** | **0.15600177** | 0.10630817 | E1′ |
| 14 | SFT baseline for R2 | ★ **validation** | 0.18314210 | 0.12011306 | — |
| 15 | GRPO-opt baseline for R2 | ★ **validation** | 0.17122683 | 0.11443492 | — |

**Grade legend** — **E1** manifest-verified · **E1′** frozen archive, hash-pinned ·
**E2** artifact present but no manifest · **E3** local artifact · **E4** artifact absent.
The grade describes **how far the raw prediction artifact can be traced**, not how much the
experiment can be trusted.

**★ Two readings that must not be missed:**
* **Rows 7, 9 and 11 are scored on the shuffled split**, because their target SIDs *are* the
  shuffled ones. Comparing a clean row with a shuffled row is meaningless — only each row's
  **own** clean→shuffled drop is interpretable.
* **Row 3 is the one arm whose number cannot be recomputed** — its prediction artifact is
  gone, so `HR@20 = 0.17626296` is only ever a *documented* figure (see
  [§7](#7-limitations--evidence)). Its **checkpoint and per-step logs do survive**, so the
  model is not lost; re-evaluating it would be a **new run**, not a recovery.

### Three headline effects

| Effect | Clean | Treated | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| **Semantic-ID intervention** (full recipe) | 0.19832341 | 0.10765497 | −0.09066844 | **−45.7175 %** HR@20 · **−30.7924 %** NDCG@20 |
| **Same intervention, sequence-only recipe** | 0.17802780 | 0.10125745 | −0.07677035 | **−43.1227 %** HR@20 · **−29.1387 %** NDCG@20 |
| **Same intervention, 1.5B backbone** | 0.19964703 | 0.11625855 | −0.08338848 | **−41.7680 %** HR@20 |

**⇒ The intervention effect survives removing the explicit text-auxiliary objectives and a
3.12× larger backbone.** Full grids are in the
[detailed record](docs/notes/experiment_summary.md).

### GRPO efficiency (the one clearly positive engineering result)

| Metric | Original 0.25 ep | Optimized 0.25 ep | Change |
|---|---|---|---|
| wall-clock | 6 111.57 s | 1 598.19 s | **−73.8 %** |
| throughput | 0.717 step/s | 2.74 step/s | **3.82×** |
| peak VRAM | 7.628 GB | 7.628 GB | unchanged |
| HR@20 | 0.17626296 | 0.17538054 | **−0.088 pp** |

Quality is **approximately preserved**, not lossless: single seed, and disabling
in-training evaluation can change RNG consumption, so identical optimization trajectories
are **not** assumed.

---

## 3. Controlled SID Intervention

> **Deep dive: [`docs/experiments/E2_shuffled_sid.md`](docs/experiments/E2_shuffled_sid.md)**
> — why a naive random SID swap cannot support attribution, the exact bucketed-derangement
> algorithm, the 18 invariants, the three swapped launcher paths, and what the result does
> and does **not** support.

### Why a naive swap is not enough

Replacing SIDs at random changes at least five things at once — the codebook, the SID
multiset, the trie, the collision structure and the popularity strata — so any drop is
unattributable. The intervention therefore changes **only which item owns which SID**.

### Design and strength

| Property | Value |
|---|---|
| catalogue items / unique SIDs | 3 686 / 3 670 |
| SID tokens | 560 (vocab 151 665 → 152 225) |
| collision-involved items | **31 intentionally frozen** (not shuffled) |
| singleton items permuted | **3 655** |
| permutation unit | whole 3-token SID codes, within train-frequency strata |
| strata | 0 / 1 / 2 / 3-5 / 6-20 / >20 by train next-item frequency |
| seed | 42, deterministic |
| singleton exact-SID retention | **0.0000 %** |
| all-item exact-SID retention | 0.8410 % (= the 31 frozen items) |
| pair-identity retention, prefix depth 1 / 2 | 2.8155 % / 0.6885 % |
| prompts / targets changed | 98.81 % / 93.01 % |

> **The 31 frozen items are retained, not removed.** They stay in the shuffled train,
> valid and test files with their original SIDs, and are trained on and evaluated like any
> other item. "Frozen" means only that their item↔SID assignment was left unshuffled —
> which is exactly what keeps the collision structure and the exact SID multiset intact.
> The 317 unchanged targets are **exactly** the rows whose target item is one of those 31
> (machine-verified, set difference 0 in both directions).

### Held invariant (18/18 checks)

Exact SID **set**, exact SID **multiset**, SID token vocabulary, trie prefix sets at depths
1/2/3, collision group membership, popularity strata, user and item identity, titles and
descriptions, sequence lengths, and the SFT recipe. The original and shuffled info files
produce **byte-identical decoding masks**, so the intervention introduces no change in
decoding behaviour.

### What this result supports — and what it does not

**✅ Supported:** Semantic SID assignment contributes substantially to the full MiniOneRec
SFT recipe, end to end (HR@20 −45.72 %), and the effect is not mainly carried by the
explicit text-auxiliary tasks (−43.12 %) nor by backbone scale (−41.77 %).

**❌ Not supported:**
* **"the model learned a complete SID semantic hierarchy"** — the shuffle destroys the
  semantic correspondence *and* the arbitrary one together, and **no experiment separates
  them**;
* **"the drop equals the share of semantic information"** — it is the sum of both;
* **"the model only memorises the table"** — after shuffling, HR@20 is still 0.10765,
  **well above** the compact SASRec baseline 0.07148 under the shared metric;
* **"the SID codebook's own construction algorithm is verified"** — see
  [§7](#7-limitations--evidence).

---

## 4. SFT Ablation & GRPO Negative Results

### 4.1 Sequence-only ablation

> **Deep dive: [`docs/experiments/E3_seq_only_ablation.md`](docs/experiments/E3_seq_only_ablation.md)**

SFT here trains **three** objectives: the main task (history SID → target SID, 36 259 rows)
plus two explicit text-auxiliary objectives — `SidItemFeatDataset` (SID ↔ title alignment,
7 316 rows) and `FusionSeqRecDataset` (history SID → target title, 36 259 rows), giving
`36 259 + 7 316 + 36 259 = 79 834`. With `--sft_mode seq_only`, the latter two are **not
instantiated at all** (the run logs a single `36259 = 36259` component).

| Arm | HR@20 clean → shuffled | Relative Δ |
|---|---|---|
| Full SFT | 0.19832341 → 0.10765497 | **−45.72 %** |
| Sequence-only | 0.17802780 → 0.10125745 | **−43.12 %** |

**★ Three precision notes:**
* What is removed is the **explicit text-auxiliary supervision**, not all textual
  information — the SIDs themselves remain text-derived. **Where the shipped P0 catalogue
  came from is not reproducible here** (see [§7](#7-limitations--evidence)).
* **This is not a single-variable isolation.** The two arms differ in the objective set
  *and* in sample count (79 834 vs 36 259), optimizer steps, and the relative warmup share.
* `FusionSeqRecDataset` **only runs its title task**: its description branch is commented
  out in the source, so **no description-supervision objective has ever trained here**, and
  its description-mapping helper is dead code under the current configuration.

### 4.2 GRPO, R1 and R2 — all negative

> **Deep dive: [`docs/experiments/E4_grpo_negative_results.md`](docs/experiments/E4_grpo_negative_results.md)**
> — the rollout and loss in code, the exact per-rank reward arithmetic for G = 16, the five
> experiment versions side by side, and the R1/R2 logic chain.

**Every GRPO arm scores below the clean SFT baseline** (rows 3–6 of the table in §2).

| experiment | change | outcome |
|---|---|---|
| **R1** ReRe-style rank reward | replaced the `ranking` reward with ReRe Eq.7–9 | zero-advantage ratio fell to **0.0** on every step, yet test HR@20 fell to **0.05912199** (**−70.19 %** vs SFT) and the output distribution collapsed to **99** distinct top-1 predictions, most-common share **33.6 %** |
| **R2** reachability-guided | exact-match 0/1 reward + a route cache putting the first GT SID token in the prompt for unreachable samples; supervised prefix CE (coef. 0.1) | zero-advantage ratio improved to **0.4750** (from ~0.70), yet **validation** HR@20 fell to **0.15600177** vs SFT 0.18314210 (**−14.82 %**) and old GRPO 0.17122683 (**−8.89 %**) |

**★ The zero-advantage ratio is neither sufficient nor a proven bottleneck.** Old GRPO
carried ~70 % zero-advantage groups and still outscored R2; R1 drove the ratio to 0.0 and
collapsed. Whether that ratio is a *long-run bottleneck* was **never tested by a
budget-matched control**, so it is recorded as **UNKNOWN**, not settled. What *is* exact:
a zero-advantage group's advantages are all zero, so it contributes **precisely zero**
gradient on that step.

**★ Mechanism notes are hypotheses, not proven causes.** Tracing the sign through advantage
and loss shows that in a no-ground-truth group the gradient **pushes down** the candidate
the model ranked first and **pushes up** the one it ranked last — an **inverse-ranking**
pressure, not a self-reinforcing one. That is *consistent with* the observed collapse, but
**no control run isolates it**. R2 changed the reward **and** the prompt routing at once and
has **no ablation separating them**, so the cause of its regression is **UNKNOWN**. (The
arithmetic behind that claim: under ReRe a no-ground-truth group has a group reward std of
exactly **0.031992**, while every ground-truth-bearing group lands in **0.267454 –
0.268875**; because the two populations separate, the logged median `group_reward_std_mean`
of 0.031992 forces at least half of R1's groups to be no-hit groups.)

In short: **SFT remains the best model in this study, and none of the RL methods beat it.**

---

## 5. Implementation & Contributions

### Upstream vs. original — what is whose

| Layer | Origin | Examples |
|---|---|---|
| **The framework** | **Upstream MiniOneRec** (unmodified in substance) | `sft.py`, `rl.py`, `data.py`, `evaluate.py`, `calc.py`, `LogitProcessor.py`, `minionerec_trainer.py`, `rq/` SID-construction, the SID catalogue, item titles/descriptions |
| **The study** | **This work** | the shuffled-SID intervention and its 18 invariants; the seq-only ablation; the GRPO/R1/R2 experiments; the compact SASRec baseline; the inference benchmark; every negative result |
| **Correctness fixes** | **This work** | reward↔sample binding by stable `sample_id` (was prompt text); stable namespaced IDs; `count_0` in `LogitProcessor.py`; SID-depth inference replacing a hardcoded depth |
| **Audit & reproducibility tooling** | **This work** | `analysis/` preflights and audits, the strict shuffled-data builder, `analysis/_paths.py`, and the provenance records under `docs/` |
| **Experiment infrastructure** | **This work** | the frozen 17 516-row GRPO subset; `rl_router.py`; `rere_reward.py`, `rl_reward.py`, `rl_prefix_ce.py`; the `patches/grpo_*.sh` launchers |

**★ Deliberately *not* claimed:** the underlying method, architecture, SID-construction
recipe, datasets and checkpoints are **upstream work**. The contribution here is the
**controlled study**, the correctness fixes and the audit trail.

### Correctness fixes and protocol corrections

| Issue found | Fix / correction |
|---|---|
| Reward matched to samples by **prompt text** (a reward could be credited to the wrong sample, making RL results unattributable) | Stable `sample_id` → target binding, fail-loud, no prompt fallback |
| `seq:{local_idx}` IDs renumbered rows after DataFrame sampling | Namespaced, stable sample IDs; split identity folded in; fixed for the list-conversion path |
| SFT and evaluation prompt text could silently drift apart | Shared `build_recommendation_prompt()` |
| GRPO degeneracy was invisible | Added per-step **zero-advantage group ratio** and KL logging |
| A hinted prompt made the constrained decoder force EOS | `count_0` counts hinted SID tokens as already generated; default 0 leaves the normal path bit-for-bit unchanged |
| **`checkpoint-4379` was mislabelled** as 1.5 epoch | It is **0.25 epoch**; `checkpoint-26274` is 1.5 epoch |
| `calc.py` NDCG looked mis-scaled | It accumulates in **natural** log and converts to log2 only at print time — the intermediate accumulator is **not** the reported metric |
| `length_penalty` looked ignored | The value reaching `model.generate()` is the CLI value (**0**); an inner helper's `1.0` default is dead code |
| A "prefix-pair retention = 100 %" metric was **wrong** | It measured group-size preservation, not pair identity; replaced |

---

## 6. Getting Started / Reproducibility

### 6.1 What you can and cannot reproduce here

**This is a research record, not a one-command reproduction.**

**★ Three things are missing from this release**, and they block different things:
**① the Amazon-derived data** (excluded — see [`DATA.md`](DATA.md)), **② the model weights**
(never shipped), **③ the historical `runs/` prediction tree** (never shipped).

**You can, on a CPU-only machine with no GPU, no weights and no data:** re-check the
**arithmetic** (`36 259 + 7 316 + 36 259 = 79 834`, and the GRPO step counts as `17 516`
prompts × epoch fraction); and re-derive the **reward and advantage arithmetic** from
`rere_reward.py`, `rl_reward.py` and `minionerec_trainer.py` — pure functions, no model
required.

**You can, once you restore the data** (see [`DATA.md`](DATA.md)): re-run the SID-construction
and intervention **audits** (`analysis/audit_shuffled_eval.py`,
`analysis/verify_info_equivalence_final.py`, `analysis/audit_trie_order.py`) and re-derive the
**intervention invariants** (`analysis/build_shuffled_sid_strict.py`).
**None of these were executed as part of preparing this release**, and this repository
does not claim otherwise.

**You cannot, even with a GPU and the data:** recompute **row 3** of the §2 table (its
prediction artifact does not exist anywhere); reproduce **R1 / R2** end to end (weights
hash-pinned but **not shipped**); recover the **LETTER** comparison (**prose only**);
re-generate the **1.5B** results from a launcher (**no 1.5B entry point ever existed**, and
none is fabricated here); or recompute any **E1′** number without extracting a frozen
records archive that this study's own governance keeps sealed.

**★ "Has a historical artifact" ≠ "recomputable from this repository."** Neither the
prediction artefacts nor the input data are part of the published tree, so nothing here is
plug-and-play.

**★ Hardware.** The runs were produced on a **single RTX 4090 (24 GB)**. The machine this
snapshot was assembled and audited on is an **8 GB laptop GPU** — enough to read, re-derive
and statically verify the evidence, **not** enough to re-run a GRPO configuration (those
peaked at 7.6–16.7 GB). **No number in this repository was validated by re-training.**

### 6.2 Environment — as the results were produced

| Item | Value |
|---|---|
| GPU | single NVIDIA RTX 4090 (24 GB) |
| Python / torch | 3.10.8 / 2.6.0+cu118 |
| transformers / trl | 4.57.1 / 0.24.0 |
| base model | Qwen2.5-0.5B (1.5B for the scaling check) |

```bash
pip install -r requirements.txt
```

**Dependency gaps found while auditing (recorded, not silently patched):**
`scikit-learn` is imported by `rl.py` but not listed; several modules under `rq/` import
`polars` / `faiss` / `loguru` / `openai`, none of which are listed or needed by any
pipeline actually run here. Neither gap was fixed, because editing the dependency files
would alter the recorded environment.

### 6.3 Entry points

> **★ This release ships WITHOUT the data.** Every command below that reads
> `data/Amazon/**` will fail until you restore that directory — see [`DATA.md`](DATA.md).
> Nothing in this repository asserts that these commands were executed here.

```bash
# Clean SFT baseline
bash sft_full.sh          # 2 epochs, effective batch 64, LR 3e-4, cutoff 512, seed 42
                          # also requires the Qwen2.5-0.5B weights (not shipped)

# Shuffled-SID intervention arm
python analysis/build_shuffled_sid_strict.py     # 1. build strict shuffled data (seed 42)
bash analysis/preflight_shuffled_sft.sh          # 2. static preflight -- refuses on inconsistency
bash scripts/sft_shuffled_sid.sh                 # 3. SFT, identical recipe
bash analysis/preflight_eval_shuffled_sid.sh     # 4. evaluation preflight
bash scripts/eval_shuffled_sid.sh                #    formal protocol, INFO_MODE=original
python analysis/audit_shuffled_eval.py           # 5. provenance audit

# Verification -- reads the historical runs/ tree, which is NOT shipped here.
python analysis/audit_predictions.py
python analysis/verify_summary_claims.py
python analysis/verify_clean_eval_metrics.py
```

**★ Requirements, in order of what blocks you first:**
1. **The Amazon-derived data** is not in this release ⇒ restore it (see [`DATA.md`](DATA.md));
2. **The model weights** are not shipped ⇒ `BASE_MODEL` must point at a local Qwen2.5-0.5B;
3. **The historical `runs/` prediction tree** is not shipped ⇒ the three verification
   scripts need it restored plus `PROJECT_ROOT` pointing at a tree that contains it.

**★ Line endings.** The shell scripts are stored with LF endings and `.gitattributes`
pins `*.sh text eol=lf`. On a clone made with `core.autocrlf=true` on Windows, a few
legacy upstream scripts can be checked out with CRLF, which makes `bash -n` (and the
script itself) fail. They pass unmodified as stored; if you see this, re-check them out
so the attribute applies:

```bash
git rm --cached -r . >/dev/null && git reset --hard
```

**★ Working directory.** The **newer launchers in `scripts/`** resolve paths from their own
location, so they are CWD-independent. **Parts of the legacy entry surface are not**: some
older scripts and tools assume the repository root as the working directory. Run these
commands from the repository root.

**★ The three verification scripts are the audit trail that produced the numbers, not a
turnkey check.**

### 6.4 Pipeline in one paragraph

Amazon Industrial_and_Scientific (3 686 items → 3 670 unique SIDs) → the upstream
pre-computed SID index is loaded and **560 SID tokens** are added (vocab 151 665 → 152 225)
→ SFT on Qwen2.5-0.5B for 2 epochs over **79 834** rows (`1 248` steps/epoch → **2 496**
steps, effective batch 64, LR 3e-4, warmup 20, bf16) → optional GRPO on a frozen
**17 516-row** subset (`seq_rec` 10 000 | title/desc 6 516 | seqtitle 1 000) → evaluation
under the formal protocol on 4 533 test samples. The interventional arm swaps the SID
assignment before SFT and then follows the same path.

**Decoding health of the formal runs:** LegalRate@20 = **100 %**, DuplicateRate@20 = **0 %**,
exactly 20 unique candidates per sample.

---

## 7. Limitations & Evidence

### 7.1 Known limitations

| # | Limitation |
|---|---|
| **L-1** | **Single seed everywhere** — no variance estimates, no significance tests; all figures are point estimates. |
| **L-2** | **One category, one dataset** (Amazon Industrial_and_Scientific). Nothing here generalises beyond it. |
| **L-3** | **Beam-restricted metric** — HR@20 is bounded by beam width 20, not full-catalogue recall. |
| **L-4** | **The 31 frozen collision items carry no treatment.** A null result cannot be attributed to them and a positive result cannot be explained by them. |
| **L-5** | **The seq-only arm is not a single-variable isolation** — objective set, sample count, optimizer steps and relative warmup share all move together. |
| **L-6** | **The 1.5B check is not a single-variable isolation** either (backbone scale plus possible memory/batching differences), and its clean gain is small (+0.1324 pp). |
| **L-7** | **SASRec is not a controlled counterpart** — it shares only the SID-level Top-K metric; generation/scoring differ, and the SFT arm trains alignment objectives SASRec has no counterpart for. |
| **L-8** | **GRPO scope** — 17 516 rows, one reward family, one configuration. This is **not** a general statement that GRPO is ineffective for recommendation. |
| **L-9** | **The shuffled arm's training was interrupted and resumed** (~step 2 180, from `checkpoint-2125`), so the two arms are **not** a bitwise-identical optimization trajectory. |
| **L-10** | **Batch-dependent prediction divergence** was observed in the inference benchmark while aggregate HR/NDCG stayed nearly stable — recorded as an observation, **not** attributed. |
| **L-11** | **Observational analyses are correlational** (popularity, prefix affinity); the matched branch-size control narrows but does not remove confounding. |
| **L-12** | In the 1.5B check the clean gain is small and **not** uniform across cutoffs (HR@3 and HR@5 are slightly *lower* than at 0.5B while HR@1 and HR@10 are higher). |

### 7.2 Open questions that remain open — and the evidence behind each

| Question / item | Status | Evidence note |
|---|---|---|
| **Which quantizer produced the shipped P0 SID catalogue?** | ★ **UNKNOWN** | `rq/` contains RQ-VAE, RQ-Kmeans, constrained RQ-Kmeans and RQ-Kmeans+ implementations, but **no codebook or quantizer checkpoint is present** and rebuilding them was deliberately excluded. **No result depends on the answer** — both arms of every comparison use the **identical** codebook. |
| **R1's collapse mechanism** | ★ **Hypothesis** | Consistent with Eq.9's within-group normalisation producing an inverse-ranking gradient; **no control run isolates it**. |
| **R2's regression cause** | ★ **UNKNOWN** | Reward and prompt routing changed at once, with **no ablation separating them**. |
| **Is the zero-advantage ratio a bottleneck?** | ★ **UNKNOWN** | Only its *sufficiency* is refuted. |
| **Row 3 of §2** | ★ **E4 — no prediction artifact** | Its HR@20 cannot be recomputed; its checkpoint and per-step logs **do** survive in the study's private evidence archive, so the model is not lost. |
| **R1 / R2 weights** | ★ **Hash-pinned, not shipped** | Metrics recomputable, runs not repeatable. |
| **LETTER comparison** | ★ **Prose only** | No prediction, metric, log or checkpoint anywhere — it must not be listed beside arms that have artifacts. |
| **1.5B entry point** | ★ **Never existed** | Only the recorded configuration and artifacts exist; none is fabricated here. |
| **`analysis/verify_*.py`** | ★ **Not self-contained** | They expect a historical `runs/` prediction tree that is **not** shipped. |

### 7.3 ★ Git history and provenance (read this before citing a commit)

> **The git history of this repository is a *synthetic migration baseline* created while
> assembling the record.** The initial commit (`090a331`) is **not** the upstream
> MiniOneRec history, and **no commit here is the revision the training runs were made at**.
> No commit should be read as a "training-time version".

The upstream framework, method, datasets and checkpoints are **upstream work** under
Apache-2.0. The study, the correctness fixes and the audit trail are this repository's
contribution. Where a number belongs to an upstream baseline rather than to this study, it
is labelled as such.

### 7.4 Data provenance and redistribution

The eval/train CSVs, the SID index, the item-metadata JSON and the info files are all
**derived from the public Amazon review data** (Ni, Li & McAuley — see the
[dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)), which asks to be
**cited** and **does not state an explicit redistribution licence**.

> **⇒ The repository's Apache-2.0 licence covers this repository's source code, not
> third-party data.**
>
> **★ Accordingly, this release ships WITHOUT any of that corpus.** The redistribution
> status of the derived files was **not confirmed**, so `data/Amazon/**` and the
> early-generation `deprecated/**` notes were **excluded from the published snapshot**.
> See **[`DATA.md`](DATA.md)** for the full list, the legal reasoning, and how to obtain
> the data if you accept its terms.
>
> **Nothing was deleted from the research record**, and **no third-party content is
> relicensed by this repository**. The exclusion affects *distribution only*: the metrics,
> conclusions and limitations reported here are unchanged.

---

## 8. Documentation & Attribution

| Document | What it covers |
|---|---|
| **[`DATA.md`](DATA.md)** | ★ **What data is excluded from this release, why, and how to obtain it** — read this before trying to run anything |
| **[`docs/ENTRYPOINTS.md`](docs/ENTRYPOINTS.md)** | ⭐ **Execution-entry layer** — every study group's real launcher, call chain, preflight, runnability and known gaps |
| **[`docs/experiments/E2_shuffled_sid.md`](docs/experiments/E2_shuffled_sid.md)** | The SID intervention: design, 18 invariants, and its exact evidential reach |
| **[`docs/experiments/E3_seq_only_ablation.md`](docs/experiments/E3_seq_only_ablation.md)** | SFT's three objectives, response-only loss, and why seq-only is not single-variable |
| **[`docs/experiments/E4_grpo_negative_results.md`](docs/experiments/E4_grpo_negative_results.md)** | GRPO rollout/loss in code, reward arithmetic, five versions, the R1/R2 logic chain |
| **[`docs/notes/experiment_summary.md`](docs/notes/experiment_summary.md)** | ⭐ **Authoritative experiment ledger** — full grids, per-stratum breakdowns, failure diagnostics |
| **[`docs/00_INDEX.md`](docs/00_INDEX.md)** · **[`docs/provenance/`](docs/provenance/)** | Index of all written records; RL post-training, reproducibility and LETTER provenance |

```
├── README.md                        this file
├── DATA.md                          ★ what data is excluded, and how to obtain it
├── docs/                            ENTRYPOINTS.md · 00_INDEX.md · experiments/ · notes/ · provenance/
├── scripts/  analysis/  baselines/  patches/  tools/
│                                    this work: launchers, audits, baselines, GRPO infra
├── sft.py  rl.py  minionerec_trainer.py  data.py       upstream pipeline
├── evaluate.py  calc.py  LogitProcessor.py             upstream evaluation
├── rq/                                                 upstream SID construction
└── data/Amazon/                     ★ NOT INCLUDED in this release -- see DATA.md
```

---

# Upstream framework & attribution

**MiniOneRec** is the first fully open-source **generative recommendation** framework:
**SID construction** → **supervised fine-tuning** → recommendation-oriented **RL**.

> **This repository is an independent reproduction and analysis of MiniOneRec, not the
> upstream project.** All upstream code is used unmodified except where listed in
> [Implementation & Contributions](#5-implementation--contributions).

| | |
|---|---|
| **Upstream repository** | https://github.com/AkaliKong/MiniOneRec |
| **Upstream paper** | [arXiv:2510.24431](https://arxiv.org/abs/2510.24431) — *MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation* |
| **Upstream models** | [Huggingface](https://huggingface.co/kkknight/MiniOneRec) · [Modelscope](https://modelscope.cn/models/k925238839/MiniOneRec) |
| **License** | Apache-2.0 (see [`LICENSE`](LICENSE)) |
| **Institutions** | LDS · AlphaLab · NExT (USTC) |
| **Acknowledgements** | [ReRe](https://github.com/sober-clever/ReRe) · [LC-Rec](https://github.com/zhengbw0324/LC-Rec) |

**Vendored code deliberately not redistributed.** `rq/letter/` is upstream LETTER source.
That repository **declares no license**, so redistribution is not permitted; only
`rq/letter/UPSTREAM_MANIFEST.json` is versioned, recording the pinned upstream commit and a
per-file SHA256 so the directory can be rebuilt and verified locally with
`tools/fetch_letter_upstream.py`.

### Citation

If you find the upstream code, paper or models helpful, please consider citing them — and
starring the upstream repository ⭐

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender},
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}
```

---

<div align="center">
Upstream framework by the MiniOneRec authors. Reproduction and analysis in this
repository by the repository owner.
</div>
