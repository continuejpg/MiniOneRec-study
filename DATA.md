# DATA.md — what data is NOT in this repository, and how to obtain it

**This repository is a curated public release snapshot of a research study, not the
original working repository and not a training-time revision.**

To keep the release free of third-party data whose redistribution terms are **not
confirmed**, the entire Amazon-derived corpus has been **deliberately excluded**.

---

## 1. Why the data is not here

The train/valid/test CSVs, the SID index, the item-metadata JSON and the info files are all
**derived from the public Amazon review / product-metadata corpus** (Ni, Li & McAuley), which
**asks to be cited but states no explicit redistribution licence**.

> **"Not forbidden" is not the same as "permitted."**
> This repository's Apache-2.0 licence covers **its own source code**, not third-party data.

This is consistent with the study's own provenance notes, which record the **raw** Amazon
CSVs as excluded **for dataset-licensing reasons** and instruct readers to download them and
rebuild. The public release simply applies the same standard to the **derived** files.

**No third-party content is relicensed by this repository.**

---

## 2. What was excluded

| Path | Files | Size | Content |
|---|---|---|---|
| `data/Amazon/{train,valid,test}/*.csv` | 6 | 52.5 MB | interaction splits carrying `user_id`, item ids and **product titles** |
| `data/Amazon/index/*.item.json` | 2 | 5.6 MB | item metadata carrying **ASINs and product titles/descriptions** |
| `data/Amazon/index/*.index.json` | 2 | 0.3 MB | Semantic-ID catalogue (SID ↔ item-id mapping) |
| `data/Amazon/info/*.txt` | 2 | 0.7 MB | `SID ⇥ title ⇥ item_id` triples used for constrained decoding |
| `deprecated/**` | 20 | 0.4 MB | early-generation notes: superseded metrics and machine-specific paths, unrelated to the study |

**Everything else ships as-is: 185 files** — the upstream framework, this study's code,
audits, launchers, and all documentation.

---

## 3. How to obtain the data

### Option A — from the upstream MiniOneRec repository (recommended)

The upstream MiniOneRec project **does track** all of these files. If you accept its
distribution of them, take the `data/` directory from there:

```bash
git clone https://github.com/AkaliKong/MiniOneRec.git /tmp/upstream
cp -r /tmp/upstream/data/Amazon  <this-repo>/data/
```

`data/Amazon/**` in the upstream repository is byte-identical to the files used in this
study, so the recorded metrics remain traceable after restoring them.

### Option B — rebuild from the original source

The upstream pipeline converts the public Amazon corpus into the split CSVs:

```
raw Amazon .inter files  ──►  convert_dataset.py  ──►  *_5_2016-10-2018-11.csv
                                                        + info/*.txt
```

* `convert_dataset.py` is present in this repository.
* The 5-core filter and the 2016-10 … 2018-11 window are encoded in the output filenames.
* ⚠ The raw `*.inter` inputs are **not** in this repository, and the SID catalogue's own
  construction algorithm is **not reproducible here** (see README §7.2). Rebuilding the
  *catalogue* would require the upstream quantizer code in `rq/` plus artifacts that are not
  shipped.

### Option C — cite and use

Whatever route you take, cite the dataset as its authors request (Ni, Li & McAuley,
*Justifying recommendations using distantly-labeled reviews and fined-grained aspects*,
EMNLP 2019).

---

## 4. What still works without the data

| Task | Needs the data? |
|---|---|
| Reading all documentation, deep dives and the experiment ledger | ❌ no |
| `bash -n` and other static checks | ❌ no |
| Re-deriving the **reward / advantage arithmetic** (`rere_reward.py`, `rl_reward.py`) | ❌ no — pure functions |
| Re-deriving the **dataset and step arithmetic** (`36 259 + 7 316 + 36 259 = 79 834`, `4 379 / 17 516 = 0.25`) | ❌ no |
| The `analysis/` **audits and preflights** | ✅ **yes** — they read the CSVs, index and info files |
| Any SFT / GRPO launcher | ✅ **yes** |
| Recomputing any headline metric | ✅ **yes** — and additionally needs the prediction artifacts, which are not published |

**⇒ With the data restored and the prediction artifacts supplied, the audits run; without
them, only the pure-function and arithmetic checks are available.** Nothing in this release
asserts that a verification was executed here.

---

## 5. Git history and the training revision

> **This repository's git history is a fresh, curated publication history.** Its first commit is
> **not** the upstream MiniOneRec history, and it is **not the revision the training runs were
> made at**.

The original working repository accumulated a mixed-source history that included the
third-party data above; that history was **deliberately not copied**. This release is a
**snapshot of the curated tree**, so:

* you cannot use `git log` or `git bisect` to reach a training-time state;
* the recorded metrics are traceable through the **documented artifacts and hashes** in
  `docs/`, not through commit ancestry.

See README §7.3 for the full provenance statement.
