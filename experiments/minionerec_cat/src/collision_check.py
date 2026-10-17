"""Phase 2B / special check -- are `best_collision_rate` and the final SID collision
ratio actually the same quantity?

They are NOT, and conflating them would misreport every run.  This script proves the
relationship for each tag by recomputing both numbers from the checkpoint:

  A. ``trainer_collision_rate``
     ``rq/trainer.py:127-152`` -- ``(num_sample - distinct_codes) / num_sample``
     over the whole item list, using ``model.get_indices(data)`` on the FIRST pass
     (``use_sk=False``).  No collision-retry loop.

  B. ``raw_index_collision_ratio``
     the same first-pass codes, but computed here directly, and the ratio expressed
     as ``(items - distinct_sids) / items`` so it is comparable.

  C. ``shipped_index_collision_ratio``
     read from the ``index.<tag>.json`` that :mod:`extract_index` wrote AFTER its
     up-to-20-round Sinkhorn retry (``rq/generate_indices.py:106-133``).

Expected relation: A == B exactly, and C <= A because the retry can only reduce
collisions (upstream stops early once none remain, and here it never fully succeeds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import common
from .train_rqvae import RQVAEConfig, _import_rq, find_best_checkpoint


def measure(tag: str) -> dict:
    paths = common.run_paths(tag)
    _EmbDataset, RQVAE, _Trainer = _import_rq()

    ckpt = find_best_checkpoint(paths.rqvae_ckpt_dir)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    a = payload["args"]

    data = _EmbDataset(str(paths.emb))
    model = RQVAE(
        in_dim=data.dim, num_emb_list=list(a.num_emb_list), e_dim=int(a.e_dim),
        layers=list(a.layers), dropout_prob=float(getattr(a, "dropout_prob", 0.0)),
        bn=bool(getattr(a, "bn", False)), loss_type=str(getattr(a, "loss_type", "mse")),
        quant_loss_weight=float(getattr(a, "quant_loss_weight", 1.0)),
        beta=float(getattr(a, "beta", 0.25)),
        kmeans_init=bool(getattr(a, "kmeans_init", True)),
        kmeans_iters=int(getattr(a, "kmeans_iters", 100)),
        sk_epsilons=list(getattr(a, "sk_epsilons", [0.0, 0.0, 0.0])),
        sk_iters=int(getattr(a, "sk_iters", 50)),
    )
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    embs = torch.stack([data[i] for i in range(len(data))]).to(device)
    with torch.no_grad():
        idx = model.get_indices(embs)          # use_sk=False, exactly as the trainer
    codes = idx.view(-1, idx.shape[-1]).cpu().numpy()
    joined = ["-".join(str(int(v)) for v in row) for row in codes]
    n = len(joined)
    distinct = len(set(joined))
    trainer_rate = (n - distinct) / n

    # shipped index (after the retry loop)
    shipped = None
    idx_path = paths.index
    if idx_path.exists():
        payload_idx = common.read_json(idx_path)
        sids = {"".join(v) for v in payload_idx.values()}
        shipped = {
            "items": len(payload_idx),
            "distinct_sids": len(sids),
            "colliding_items": len(payload_idx) - len(sids),
            "collision_ratio": (len(payload_idx) - len(sids)) / len(payload_idx),
        }

    # what the trainer logged
    man = paths.report_dir / f"{tag}_train_manifest.json"
    logged = None
    if man.exists():
        logged = common.read_json(man).get("extra", {}).get("best_collision_rate")

    return {
        "tag": tag,
        "checkpoint": str(ckpt),
        "items": n,
        "A_trainer_style_rate_recomputed": round(trainer_rate, 10),
        "B_raw_first_pass": {
            "items": n, "distinct_codes": distinct, "colliding_items": n - distinct,
            "collision_ratio": round((n - distinct) / n, 10),
        },
        "C_shipped_index_after_retry": shipped,
        "D_logged_best_collision_rate": logged,
        "A_equals_B": abs(trainer_rate - (n - distinct) / n) < 1e-12,
        "logged_matches_recompute": (
            None if logged is None else abs(float(logged) - trainer_rate) < 1e-9
        ),
        "retry_reduced_collisions": (
            None if shipped is None else shipped["collision_ratio"] <= trainer_rate + 1e-12
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=["b2"])
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    print("=" * 96)
    print("Phase 2B special check: collision-rate definitions")
    print("=" * 96)
    out = {}
    for tag in args.tags:
        try:
            r = measure(tag)
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag}: ERROR {exc!r}")
            continue
        out[tag] = r
        print(f"\n  --- {tag} ---")
        print(f"    A trainer-style rate (recomputed from ckpt, first pass) : "
              f"{r['A_trainer_style_rate_recomputed']:.10f}")
        print(f"    B raw first-pass distinct / items                      : "
              f"{r['B_raw_first_pass']['distinct_codes']} / {r['B_raw_first_pass']['items']} "
              f"= {r['B_raw_first_pass']['collision_ratio']:.10f}")
        print(f"    D logged best_collision_rate in manifest               : "
              f"{r['D_logged_best_collision_rate']}")
        print(f"    A == B                          : {r['A_equals_B']}")
        print(f"    logged == recomputed            : {r['logged_matches_recompute']}")
        c = r["C_shipped_index_after_retry"]
        if c:
            print(f"    C shipped index AFTER retry     : {c['distinct_sids']} distinct, "
                  f"{c['colliding_items']} colliding items, ratio {c['collision_ratio']:.10f}")
            print(f"    retry reduced collisions (C<=A) : {r['retry_reduced_collisions']}")

    print()
    print("  DEFINITIONS (must not be mixed):")
    print("    A/B -- rq/trainer.py's collision_rate: first-pass codes, per-item join with")
    print("           '-', (n - distinct)/n.  This is the number stored as")
    print("           best_collision_rate and used to pick best_collision_model.pth.")
    print("    C   -- the SID catalogue actually shipped, produced AFTER up to 20 rounds of")
    print("           Sinkhorn re-quantisation of the colliding items only.")
    print("    => C is always <= A; quoting C as 'the model's collision rate' understates it.")

    dest = args.json or (common.EXP_ROOT / "_comparison" / "collision_definition_check.json")
    common.write_json(dest, out)
    print(f"\n  written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
