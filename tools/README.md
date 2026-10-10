# tools/ — reproducibility and audit scripts

Everything here is a **script**. No binary artifact, checkpoint, dataset or
prediction file is tracked; see `.gitignore` (`artifacts/`, `runs/`).

Naming history is preserved on purpose: `stage1_*`, `stage2_*`, `stage25_*`,
`stage3_*`, `stage4_*` refer to the order the work was actually done, so an
external reviewer can follow the provenance rather than a tidied-up fiction.

---

## A. Canonical entry points (start here)

Run everything from the repository root. `$D` = repo root on the machine that ran
the experiments.

| # | Purpose | Canonical script |
|---|---|---|
| 1 | **Stage 1 — text embedding** (Qwen2.5-0.5B masked-mean, `[3686, 896]`) | `tools/letter_stage1/stage1_text_embedding.py` |
| 2 | **Stage 1 — collaborative embedding** (SASRec-32, `[3686, 32]`) | `tools/letter_stage1/stage1_sasrec32_embedding.py` |
| 3 | **Stage 1 — item manifest** (id space + text provenance) | `tools/letter_stage1/stage1_build_item_manifest.py` |
| 4 | **Stage 2 — LETTER tokenizer training** (5000 epochs, 4-level) | `tools/letter_stage1/stage2_train_letter.py` |
| 5 | **Stage 2 — SID generation** (official sinkhorn anti-collision loop) | `tools/letter_stage1/stage2_generate_sid.py` |
| 6 | **Stage 4 — content-only SID generation** (`alpha=beta=0`) | `tools/letter_stage1/stage4_generate_sid.py` |
| 7 | **Remap** P0 splits to any SID index (byte-faithful) | `tools/letter_stage1/stage3_step34_remap.py` |
| 8 | **Tokenizer construction** from pristine base + resize assertion | `tools/letter_stage1/stage3_step1_tokenizer.py` |
| 9 | **SFT preflight** (depth/trie/dataset/forward-backward smoke) | `tools/letter_stage1/stage3_step5_preflight.py` |
| 10 | **Depth + trie regression** for both catalogues | `tools/letter_stage1/stage25_verify_sid_depth.py` |
| 11 | **SFT launcher** (P0 recipe) | `sft_letter_full.sh`, `sft_content_only_full.sh` |
| 12 | **Eval launcher** (4553 samples, beam 20, constrained) | `eval_letter.sh`, `eval_content_only.sh` |
| 13 | **Eval legality / duplication audit** | `tools/letter_stage1/stage3_step8_legality.py` |
| 14 | **Three-way comparison table** | `tools/letter_stage1/stage4_step13_compare.py` |
| 15 | **Unit / regression test** (CPU only, no model) | `tools/tests/test_sid_depth.py` |

### Typical chain

```bash
# Stage 1
python tools/letter_stage1/stage1_build_item_manifest.py
python tools/letter_stage1/stage1_text_embedding.py
python tools/letter_stage1/stage1_sasrec32_embedding.py

# Stage 2  (Treatment T)
python tools/letter_stage1/stage2_train_letter.py --epochs 5000 \
    --out-dir artifacts/letter_stage2
python tools/letter_stage1/stage2_generate_sid.py \
    --ckpt artifacts/letter_stage2_5k/letter_tokenizer_best_collision.pth \
    --out-dir artifacts/letter_stage2

# Stage 4  (Content-only B) -- only alpha and beta differ
python tools/letter_stage1/stage2_train_letter.py --epochs 5000 \
    --alpha 0.0 --beta 0.0 --out-dir artifacts/letter_stage4_content_only
python tools/letter_stage1/stage4_generate_sid.py

# Remap + tokenizer + preflight, per variant
python tools/letter_stage1/stage3_step34_remap.py \
    --letter-index artifacts/letter_stage4_content_only/content_only_index.json \
    --out-root     artifacts/letter_stage4_content_only/data
python tools/letter_stage1/stage3_step1_tokenizer.py \
    /path/to/Qwen2.5-0.5B \
    artifacts/letter_stage4_content_only/content_only_index.json \
    artifacts/letter_stage4_content_only
PREFLIGHT_STAGE=4 python tools/letter_stage1/stage3_step5_preflight.py

# SFT + eval
bash sft_content_only_full.sh
bash eval_content_only.sh

# Regression (CPU, seconds)
python tools/tests/test_sid_depth.py
```

---

## B. Audit / diagnostic scripts (kept for provenance, not entry points)

These were used to *establish* the results. Several of them exist because an
earlier hypothesis was wrong and had to be falsified. They are retained so the
negative evidence is auditable.

| Script | What it establishes |
|---|---|
| `stage1_diag_duplicates.py` | the 11 duplicate text-embedding rows come from 11 byte-identical cleaned texts, not a bug |
| `stage1_final_audit.py` | Stage 1 artifact audit (27 checks) |
| `stage2_measure_cf_loss.py` | the **true** collaborative loss, measured independently of the trainer's log slots |
| `stage2_cf_baseline_test.py` | the CF objective is at chance level after 200 epochs and clearly learned after 5000 |
| `stage2_diag_collapse.py` | 200-epoch codebook collapse is a training-length problem, not a circular-loss bug |
| `stage2_diag_keys.py`, `stage2_diag_prefix_index.py`, `stage2_diag_prefix_index2.py`, `stage2_diag_reachability.py`, `stage2_diag_realprompt.py`, `stage2_diag_real_dataset.py`, `stage2_final_gen_probe.py`, `stage2_real_generation_test.py` | successive (partly wrong) attempts to pin down `prefix_index`; the correct semantics were finally settled by exhaustive key matching |
| `stage2_env_sanity.py` | the `k-means-constrained` install upgraded numpy to 2.x; existing modules still import |
| `stage2_vendor_patch.py` | applies the M1–M5 local edits to the vendored LETTER sources |
| `stage2_smoke_4level.py` | 4-level smoke on the token-extender + trie |
| `stage2_diag_trie_correctness.py` | trie correctness sweep |
| `stage3_probe_siditemfeat.py` | `SidItemFeatDataset` size differs per catalogue (title collisions), not a data bug |
| `stage25_verify_sid_depth.py` | depth/trie verification on the **fixed tree** (superseded by the regression test, kept for comparison) |
| `stage25_*` (patch / probe / measure / blocker) | the Stage 2.5 changes and the vocabulary-overflow diagnosis |
| `stage4_step2_grad_proof.py` | perturbation proof that `alpha=0` / `beta=0` exclude both regularisers from the gradient |
| `stage25_fix_eol.py` | restores CRLF after a patch rewrote three sources as LF |

`sitecustomize.py` is a **runtime guard**, not an analysis script: it stubs
`wandb`, which cannot be imported after the `protobuf` upgrade (see
`docs/letter_sid_experiment.md` § Limitation L4). Put its directory on
`PYTHONPATH` to activate.

---

## C. Not a recommended entry point

`tools/tests/stage5_debug_test.py` is a throwaway probe used while fixing the
regression-test fixture. Use `tools/tests/test_sid_depth.py` instead.

---

## D. Vendored upstream code is NOT in this tree

`rq/letter/` is upstream LETTER source and is **deliberately absent from version
control**: the upstream repository declares no license, so redistribution is not
permitted. See `docs/letter_sid_experiment.md` § Upstream provenance and
`tools/fetch_letter_upstream.py`, which reconstructs the directory from the pinned
upstream commit and verifies every file hash.
