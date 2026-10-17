"""MiniOneRec-CAT / audit_sid -- quantify what a SID catalogue actually looks like.

Purpose
-------
Compare SID catalogues produced by different quantiser configurations on
properties that are **measurable from the catalogue itself**, without any
recommendation experiment:

* how many codewords each layer actually uses, and how evenly
  (entropy, normalised entropy, largest-cluster share);
* how many items collapse onto a shared SID, and how many become
  **unscorable** under the evaluator's actual mapping rule;
* whether the quantiser preserved item-text neighbourhood structure, measured as
  the overlap between neighbourhoods in the original embedding space and in SID
  Hamming space;
* whether codewords are semantically coherent, measured as the ratio of
  within-codeword to across-codeword mean distance in the text embedding space.

What it deliberately does NOT do
--------------------------------
It reports no HR / NDCG and makes no recommendation-quality claim.  Higher
codeword utilisation is a property of the codebook, not evidence of better
recommendations.  Any such claim requires the SFT + evaluation arm.

Usage
-----
    python -m experiments.minionerec_cat.src.audit_sid --tag b0
    python -m experiments.minionerec_cat.src.audit_sid --tag b0 --baseline p0
"""

from __future__ import annotations

import argparse
import io
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import common
from .rebuild_derived import load_index, sid_string


# --------------------------------------------------------------------------- #
# Catalogue-level statistics
# --------------------------------------------------------------------------- #

def layer_tokens(index: dict[str, list[str]]) -> dict[str, list[int]]:
    """layer letter -> list of code values, one per item, in item-id order."""
    out: dict[str, list[int]] = defaultdict(list)
    for item in sorted(index, key=int):
        for token in index[item]:
            letter, _, value = token[1:-1].partition("_")
            out[letter].append(int(value))
    return dict(out)


def space_neighbour_overlap(emb_path: Path, index: dict[str, list[str]], *,
                            sample: int = 800, seed: int = 0,
                            ks: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Fraction of SID-Hamming top-K neighbours that are also top-K in an arbitrary space.

    Used for two distinct questions:

    * ``text``  -- did quantisation preserve **item-text** neighbourhood structure?
    * ``collab`` -- did it preserve **collaborative** neighbourhood structure?

    A caveat that must travel with the ``collab`` number: the B2 arm optimises a
    collaborative objective directly, so a higher ``collab`` overlap is partly a
    *restatement of the training objective*, not independent evidence of better
    recommendation.  It is reported as a mechanistic check, never as a benefit.
    """
    emb = np.load(emb_path).astype(np.float32)
    n = emb.shape[0]
    if n != len(index):
        raise ValueError(f"{emb_path} has {n} rows, index has {len(index)}")

    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = emb / norm
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)

    rng = np.random.default_rng(seed)
    take = min(sample, n)
    idx = np.sort(rng.choice(n, size=take, replace=False))

    tokens = layer_tokens(index)
    letters = sorted(tokens)
    codes = np.stack([np.asarray(tokens[l], dtype=np.int32) for l in letters], axis=1)
    sub_codes = codes[idx]
    hamming = (sub_codes[:, None, :] != sub_codes[None, :, :]).sum(axis=2)
    sub_sim = sim[np.ix_(idx, idx)]

    out: dict[str, float] = {}
    for k in ks:
        if k >= take:
            continue
        space_nb = np.argpartition(-sub_sim, k, axis=1)[:, :k]
        sid_nb = np.argpartition(hamming, k, axis=1)[:, :k]
        hits = sum(len(set(space_nb[r].tolist()) & set(sid_nb[r].tolist())) for r in range(take))
        out[f"overlap@{k}"] = round(hits / (take * k), 6)
    return {"embedding": str(emb_path), "sampled_items": int(take),
            "sample_seed": seed, "neighbour_overlap": out}


def entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def describe_layers(index: dict[str, list[str]], *, codebook_size: int = 256) -> dict:
    """Per-layer occupancy and distribution statistics."""
    tokens = layer_tokens(index)
    out: dict[str, dict] = {}
    for letter in sorted(tokens):
        codes = tokens[letter]
        counts = Counter(codes)
        h = entropy_bits(counts)
        h_max = math.log2(codebook_size) if codebook_size > 1 else 0.0
        top_code, top_n = counts.most_common(1)[0]
        out[letter] = {
            "items": len(codes),
            "distinct_codes_used": len(counts),
            "codebook_size": codebook_size,
            "occupancy": round(len(counts) / codebook_size, 6),
            "entropy_bits": round(h, 6),
            "max_entropy_bits": round(h_max, 6),
            "normalised_entropy": round(h / h_max, 6) if h_max else None,
            "largest_cluster_code": f"{letter}_{top_code}",
            "largest_cluster_size": top_n,
            "largest_cluster_share": round(top_n / len(codes), 6),
            "min_cluster_size": min(counts.values()),
            "mean_cluster_size": round(len(codes) / len(counts), 4),
            "codes_never_used": codebook_size - len(counts),
        }
    return out


def describe_collisions(index: dict[str, list[str]]) -> dict:
    """Collision structure including the evaluator's actual resolution rule."""
    sid2items: dict[str, list[int]] = defaultdict(list)
    for item, tokens in index.items():
        sid2items[sid_string(tokens)].append(int(item))
    for v in sid2items.values():
        v.sort()

    groups = {s: v for s, v in sid2items.items() if len(v) > 1}
    involved = sum(len(v) for v in groups.values())
    # calc.py:24-30 keeps the first (lowest) item id per SID, so every other
    # member of a collision group can never be produced by the evaluator.
    unscorable = sorted(i for v in groups.values() for i in v[1:])

    return {
        "items": len(index),
        "distinct_sids": len(sid2items),
        "colliding_items": len(index) - len(sid2items),
        "colliding_groups": len(groups),
        "items_in_collision_groups": involved,
        "unscorable_items": len(unscorable),
        "unscorable_item_ids": unscorable,
        "largest_group": max((len(v) for v in groups.values()), default=1),
        "group_size_histogram": dict(sorted(Counter(len(v) for v in groups.values()).items())),
        "groups": [
            {"sid": s, "items": v} for s, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        ],
    }


def describe_sid_lengths(index: dict[str, list[str]]) -> dict:
    lens = Counter(len(v) for v in index.values())
    return {
        "length_histogram": dict(sorted(lens.items())),
        "all_three_levels": set(lens) == {3},
        "levels": sorted(lens)[0] if len(lens) == 1 else None,
    }


# --------------------------------------------------------------------------- #
# Text-space coherence and neighbourhood preservation
# --------------------------------------------------------------------------- #

def text_similarity_stats(emb_path: Path, index: dict[str, list[str]], *, sample: int = 600,
                          seed: int = 0) -> dict:
    """Codeword coherence and neighbourhood overlap, using the embedding actually quantised.

    Two measurements, both from the text embedding space:

    ``within_vs_across``
        For each layer, the mean pairwise cosine distance among items sharing a
        codeword versus items in different codewords.  A useful layer has
        within < across.

    ``neighbour_overlap``
        Restrict to a random sample of items (the full 3686 x 3686 matrix is
        cheap, but the sample keeps the report size predictable).  For each
        sampled item take its top-K text neighbours and its top-K SID-Hamming
        neighbours, and report the mean |intersection| / K for K in {5, 10, 20}.
        This is the direct measure of "did quantisation preserve what was
        textually near".
    """
    emb = np.load(emb_path).astype(np.float32)
    n = emb.shape[0]
    if n != len(index):
        raise ValueError(f"embedding has {n} rows but the index has {len(index)} items")

    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = emb / norm
    sim = unit @ unit.T
    # Mask the diagonal AFTER computing distances, never on ``sim``: setting the
    # self-similarity to -inf would make the self-distance +inf and silently
    # poison every mean that touches it.
    np.fill_diagonal(sim, 0.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.nan)   # excluded explicitly, never averaged in

    rng = np.random.default_rng(seed)
    take = min(sample, n)
    idx = np.sort(rng.choice(n, size=take, replace=False))

    tokens = layer_tokens(index)
    letters = sorted(tokens)
    codes = np.stack([np.asarray(tokens[l], dtype=np.int32) for l in letters], axis=1)

    # ---- codeword coherence per layer ------------------------------------- #
    coherence: dict[str, dict] = {}
    for j, letter in enumerate(letters):
        col = codes[:, j]
        sub = dist[np.ix_(idx, idx)]
        same = col[idx][:, None] == col[idx][None, :]
        # Drop the diagonal from both sides: an item is not its own pair.
        off_diag = ~np.eye(take, dtype=bool)
        same = same & off_diag
        across = (~(col[idx][:, None] == col[idx][None, :])) & off_diag
        within_vals = sub[same]
        across_vals = sub[across]
        within_vals = within_vals[np.isfinite(within_vals)]
        across_vals = across_vals[np.isfinite(across_vals)]
        w_mean = float(within_vals.mean()) if within_vals.size else None
        a_mean = float(across_vals.mean()) if across_vals.size else None
        coherence[letter] = {
            "mean_within_codeword_distance": None if w_mean is None else round(w_mean, 6),
            "mean_across_codeword_distance": None if a_mean is None else round(a_mean, 6),
            "within_pairs": int(within_vals.size),
            "across_pairs": int(across_vals.size),
            "coherent": bool(w_mean is not None and a_mean is not None and w_mean < a_mean),
        }

    # ---- SID Hamming neighbours ------------------------------------------- #
    hamming = (codes[idx][:, None, :] != codes[idx][None, :, :]).sum(axis=2)

    overlap: dict[str, float] = {}
    for k in (5, 10, 20):
        if k >= take:
            continue
        text_nb = np.argpartition(-sim[np.ix_(idx, idx)], k, axis=1)[:, :k]
        sid_nb = np.argpartition(hamming, k, axis=1)[:, :k]
        hits = 0
        for r in range(take):
            hits += len(set(text_nb[r].tolist()) & set(sid_nb[r].tolist()))
        overlap[f"overlap@{k}"] = round(hits / (take * k), 6)

    return {
        "embedding": str(emb_path),
        "sampled_items": int(take),
        "sample_seed": seed,
        "codeword_coherence": coherence,
        "neighbour_overlap": overlap,
    }


# --------------------------------------------------------------------------- #
# Training-log statistics
# --------------------------------------------------------------------------- #

_LOSS_RE = None


def parse_train_log(path: Path) -> dict:
    """Extract the loss curve from the trainer's log, if we can find one.

    ``rq/trainer.py`` writes through ``logging``; the exact format depends on the
    handler, so this is best-effort and reports what it found rather than
    inventing values.
    """
    import re
    loss_re = re.compile(r"train loss:\s*([0-9.]+).*?reconstruction loss:\s*([0-9.]+)", re.I)
    epoch_re = re.compile(r"epoch\s+(\d+)\s+training", re.I)
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    if not path or not path.exists():
        return {"log": None, "found": False}

    raw = io.open(path, encoding="utf-8", errors="replace").read()
    # ``rq/trainer.py`` logs through ``set_color``, which wraps each label in ANSI
    # escapes (e.g. ``\x1b[1;34mtrain loss\x1b[0m``).  The escapes sit *between*
    # "train loss" and its colon, so the pattern only matches after they are
    # removed.  Without this the loss curve silently parses as empty.
    text = ansi_re.sub("", raw)

    losses: list[tuple[int, float, float]] = []
    for line in text.splitlines():
        m = loss_re.search(line)
        if not m:
            continue
        em = epoch_re.search(line)
        if em:
            losses.append((int(em.group(1)), float(m.group(1)), float(m.group(2))))
    if not losses:
        for m in re.finditer(
            r"epoch\s+(\d+)\s+training.{0,200}?train loss:\s*([0-9.]+)"
            r".{0,200}?reconstruction loss:\s*([0-9.]+)",
            text, re.I | re.S,
        ):
            losses.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    return {
        "log": str(path),
        "found": bool(losses),
        "epochs_logged": len(losses),
        "first": {"epoch": losses[0][0], "total": losses[0][1], "recon": losses[0][2]} if losses else None,
        "last": {"epoch": losses[-1][0], "total": losses[-1][1], "recon": losses[-1][2]} if losses else None,
        "min_total_loss": min((l[1] for l in losses), default=None),
        "curve_tail": [{"epoch": e, "total": t, "recon": r} for e, t, r in losses[-5:]],
    }


# --------------------------------------------------------------------------- #

def audit(tag: str, *, index_path: Path | None = None, emb_path: Path | None = None,
          baseline: str | None = "p0", codebook_size: int = 256,
          coherence_sample: int = 600, coherence_seed: int = 0,
          collab_embedding: Path | None = None) -> dict:
    paths = common.run_paths(tag)
    src_index = Path(index_path) if index_path else paths.index
    src_emb = Path(emb_path) if emb_path else paths.emb

    index = load_index(src_index)
    report: dict = {
        "tag": tag,
        "index": str(src_index),
        "index_sha256": common.sha256_file(src_index),
    }

    print("=" * 88)
    print(f"MiniOneRec-CAT / {tag}  -- SID quality audit")
    print("=" * 88)
    print(f"  index : {src_index}")
    print(f"  items : {len(index):,}")

    report["layers"] = describe_layers(index, codebook_size=codebook_size)
    print("  layer occupancy / entropy")
    for letter, s in report["layers"].items():
        print(f"    {letter}: used={s['distinct_codes_used']:>3}/{s['codebook_size']} "
              f"({s['occupancy']*100:5.1f}%)  H={s['entropy_bits']:.4f}/{s['max_entropy_bits']:.1f} "
              f"eff={s['normalised_entropy']:.4f}  largest={s['largest_cluster_code']}"
              f"({s['largest_cluster_size']})")

    report["collisions"] = describe_collisions(index)
    c = report["collisions"]
    print(f"  SIDs  : distinct={c['distinct_sids']:,}  colliding_groups={c['colliding_groups']}  "
          f"items_in_groups={c['items_in_collision_groups']}  unscorable={c['unscorable_items']}")

    report["sid_lengths"] = describe_sid_lengths(index)
    print(f"  levels: {report['sid_lengths']['length_histogram']}  all_three={report['sid_lengths']['all_three_levels']}")

    if src_emb.exists():
        try:
            report["text_coherence"] = text_similarity_stats(
                src_emb, index, sample=coherence_sample, seed=coherence_seed)
            tc = report["text_coherence"]
            print("  codeword coherence (within vs across, lower is better for 'within')")
            for letter, s in tc["codeword_coherence"].items():
                print(f"    {letter}: within={s['mean_within_codeword_distance']} "
                      f"across={s['mean_across_codeword_distance']} coherent={s['coherent']}")
            print(f"  neighbour overlap: {tc['neighbour_overlap']}")
            # Collaborative-neighbourhood preservation, when a cf table is available
            # for this run.  Mechanistic check only: the B2 objective targets this
            # space, so it is not independent evidence of recommendation quality.
            cf = Path(collab_embedding) if collab_embedding \
                else paths.root / "collab" / f"cf_embedding_{tag}.npy"
            if cf.exists():
                report["collab_neighbour"] = space_neighbour_overlap(
                    cf, index, sample=coherence_sample, seed=coherence_seed)
                report["collab_neighbour"]["source"] = str(cf)
                print(f"  collab neighbour overlap: "
                      f"{report['collab_neighbour']['neighbour_overlap']}")
                print(f"    source: {cf}")
                print("    (mechanistic: B2's objective targets this space; NOT a quality claim)")
        except Exception as exc:  # noqa: BLE001
            report["text_coherence"] = {"error": repr(exc)}
            print(f"  text coherence: ERROR {exc!r}")
    else:
        report["text_coherence"] = {"skipped": f"no embedding at {src_emb}"}
        print(f"  text coherence: skipped (no embedding at {src_emb})")

    if baseline:
        base_path = paths.p0_index if baseline == "p0" else Path(baseline)
        if base_path.exists():
            base = load_index(base_path)
            base_layers = describe_layers(base, codebook_size=codebook_size)
            base_coll = describe_collisions(base)
            report["baseline"] = {
                "name": baseline,
                "index": str(base_path),
                "layers": base_layers,
                "collisions": base_coll,
                "delta": {
                    "distinct_sids": c["distinct_sids"] - base_coll["distinct_sids"],
                    "colliding_groups": c["colliding_groups"] - base_coll["colliding_groups"],
                    "unscorable_items": c["unscorable_items"] - base_coll["unscorable_items"],
                    "layer_occupancy": {
                        L: report["layers"][L]["distinct_codes_used"]
                           - base_layers[L]["distinct_codes_used"]
                        for L in sorted(set(report["layers"]) & set(base_layers))
                    },
                    "layer_entropy_bits": {
                        L: round(report["layers"][L]["entropy_bits"]
                                 - base_layers[L]["entropy_bits"], 6)
                        for L in sorted(set(report["layers"]) & set(base_layers))
                    },
                },
            }
            d = report["baseline"]["delta"]
            print(f"  vs {baseline}: dDistinct={d['distinct_sids']:+d} "
                  f"dGroups={d['colliding_groups']:+d} dUnscorable={d['unscorable_items']:+d} "
                  f"dOccupancy={d['layer_occupancy']} dEntropy={d['layer_entropy_bits']}")

    # Training artefacts, if this tag has them.
    train_log = paths.log_dir / "b0_train.log"
    report["training"] = parse_train_log(train_log)
    if report["training"].get("found"):
        t = report["training"]
        print(f"  training: epochs_logged={t['epochs_logged']} min_total_loss={t['min_total_loss']}")
    manifest = paths.report_dir / f"{tag}_train_manifest.json"
    if manifest.exists():
        m = common.read_json(manifest)
        report["train_manifest"] = {
            "best_loss": m.get("extra", {}).get("best_loss"),
            "best_collision_rate": m.get("extra", {}).get("best_collision_rate"),
            "input_dim": m.get("extra", {}).get("input_dim"),
            "n_items": m.get("extra", {}).get("n_items"),
            "config": m.get("config"),
            "checkpoint": list(m.get("outputs", {}).keys()),
        }

    report["disclaimer"] = (
        "These are properties of the Semantic ID catalogue.  Codebook occupancy or "
        "entropy is NOT evidence of recommendation quality; that requires the SFT and "
        "evaluation arms, which this audit does not run."
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="b0")
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--emb", type=Path, default=None)
    p.add_argument("--baseline", default="p0", help="'p0', a path, or '' to skip")
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--coherence-sample", type=int, default=600)
    p.add_argument("--collab-embedding", type=Path, default=None,
                   help="frozen cf table to measure collaborative-neighbourhood "
                        "preservation against; defaults to this run's own table. All "
                        "B2 arms share one table, so pass it to compare like for like.")
    p.add_argument("--json", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(
        args.tag,
        index_path=args.index,
        emb_path=args.emb,
        baseline=args.baseline or None,
        codebook_size=args.codebook_size,
        coherence_sample=args.coherence_sample,
        collab_embedding=args.collab_embedding,
    )
    paths = common.run_paths(args.tag)
    paths.ensure_dirs()
    out = args.json or (paths.report_dir / f"{args.tag}_sid_quality_audit.json")
    common.write_json(out, report)
    print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
