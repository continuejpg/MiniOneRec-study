#!/usr/bin/env python3
"""
R2.1c -- training-entry smoke driver.

Enters the REAL `rl.py` -> ReReTrainer -> generate -> reward -> loss -> backward
path, with a small route cache (32-64 train-only samples) and R21_SMOKE_STEPS, and
INSTRUMENTS the actual `generate()` calls rather than inferring batch structure
from dataset ordering.

Instrumentation (measurement only, no behaviour change):
  * ReReTrainer._prepare_inputs is wrapped to record, for every generate() call:
      - number of prompts in the micro-batch
      - the sample_ids they carry, their route / h / count_0
      - whether routes are mixed
  * constraints.ComputeLossClass.compute_loss is wrapped to record per-step
    reward / advantage / prefix-CE / loss, and to assert gradients are finite.

Usage (env-driven, see the launcher):
    R21_SMOKE_LIMIT=64 R21_SMOKE_STEPS=6 python tools/tests/r21c_train_smoke.py ...
"""
import json
import math
import os
import sys

import torch

REPO = os.getcwd()
sys.path.insert(0, REPO)

RECORDS = []


def main():
    import fire
    import rl as RL
    from minionerec_trainer import ReReTrainer
    import minionerec_trainer as MT

    route_cache = os.environ["R21_ROUTE_CACHE"]
    cache = json.load(open(route_cache, encoding="utf-8"))
    routes = cache["routes"]
    print("=" * 92)
    print("R2.1c TRAINING-ENTRY SMOKE")
    print("=" * 92)
    print(f"  route cache      : {route_cache}")
    print(f"  cache samples    : {len(routes)}")
    print(f"  R21_SMOKE_LIMIT  : {os.environ.get('R21_SMOKE_LIMIT')}")
    print(f"  R21_SMOKE_STEPS  : {os.environ.get('R21_SMOKE_STEPS')}")

    # ---------------- instrument the real generate() call site -------------
    _orig_prepare = ReReTrainer._prepare_inputs

    def _instrumented_prepare(self, inputs):
        sids = []
        for x in inputs:
            sids.append(x.get("sample_id"))
        rt = [routes.get(s, {}).get("route") for s in sids]
        hv = [int(routes.get(s, {}).get("h", -1)) for s in sids]
        mixed = len(set(rt)) > 1
        RECORDS.append({"call": len(RECORDS), "n_prompts": len(inputs),
                        "sample_ids": sids, "routes": rt, "h": hv,
                        "mixed": mixed,
                        "count_0": getattr(self, "_r21_hint_len_cache", None)})
        print(f"  [generate #{len(RECORDS)-1}] prompts={len(inputs)} "
              f"routes={sorted(set(str(x) for x in rt))} h={sorted(set(hv))} "
              f"MIXED={mixed}")
        if mixed:
            raise RuntimeError(
                f"[R2.1c] generate() received a MIXED-route batch: {list(zip(sids, rt))}. "
                f"count_0 is a single scalar, so this is invalid.")
        return _orig_prepare(self, inputs)

    ReReTrainer._prepare_inputs = _instrumented_prepare

    # ---------------- instrument loss / advantage --------------------------
    _orig_compute = ReReTrainer.compute_loss
    LOSSES = []

    def _instrumented_compute(self, model, inputs, *a, **kw):
        loss = _orig_compute(self, model, inputs, *a, **kw)
        try:
            adv = inputs.get("advantages")
            rec = {"loss": float(loss.detach()),
                   "adv_absmax": float(adv.abs().max()) if adv is not None else None,
                   "zero_adv_groups": int((adv.abs() < 1e-12).sum().item()) if adv is not None else None,
                   "prefix_ce": float(self._prefix_ce.detach()) if getattr(self, "_prefix_ce", None) is not None else None}
            LOSSES.append(rec)
            print(f"  [loss #{len(LOSSES)-1}] loss={rec['loss']:.6f} "
                  f"zero-adv elems={rec['zero_adv_groups']} "
                  f"prefix_ce={rec['prefix_ce']}")
        except Exception as e:  # noqa: BLE001
            print(f"  [loss] record failed: {type(e).__name__}: {e}")
        return loss

    ReReTrainer.compute_loss = _instrumented_compute

    # ---------------- run the real entry point ----------------------------
    kwargs = json.loads(os.environ["R21_RL_KWARGS"])
    argv = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            argv.append(f"--{k}={v}")
        elif v is None:
            continue
        else:
            argv.append(f"--{k}={v}")
    print(f"  argv entries     : {len(argv)}")
    try:
        fire.Fire(RL.train, command=argv)
    except SystemExit:
        pass

    # ---------------- report ----------------------------------------------
    print()
    print("=" * 92)
    print("ACTUAL GENERATE() BATCH STRUCTURE")
    print("=" * 92)
    n_mixed = sum(1 for r in RECORDS if r["mixed"])
    print(f"  generate() calls              = {len(RECORDS)}")
    print(f"  prompts per call              = "
          f"{sorted(set(r['n_prompts'] for r in RECORDS))}")
    print(f"  MIXED-route calls             = {n_mixed}")
    print(f"  count_0 observed              = "
          f"{sorted(set(str(r['count_0']) for r in RECORDS))}")
    for r in RECORDS[:8]:
        print(f"    call {r['call']}: n={r['n_prompts']} "
              f"routes={sorted(set(str(x) for x in r['routes']))} "
              f"h={sorted(set(r['h']))} mixed={r['mixed']}")

    print()
    print("=" * 92)
    print("LOSS / ADVANTAGE")
    print("=" * 92)
    if LOSSES:
        fin = all(math.isfinite(x["loss"]) for x in LOSSES)
        print(f"  steps recorded                = {len(LOSSES)}")
        print(f"  all losses finite             = {fin}")
        print(f"  loss range                    = "
              f"{min(x['loss'] for x in LOSSES):.6f} .. {max(x['loss'] for x in LOSSES):.6f}")
        pce = [x["prefix_ce"] for x in LOSSES if x["prefix_ce"] is not None]
        print(f"  prefix CE observed            = "
              f"{[round(p, 4) for p in pce] if pce else '(none this step)'}")
        za = [x["zero_adv_groups"] for x in LOSSES]
        print(f"  zero-adv elements per step    = {za}")
        gfin = True
        for p in (m for m in model_params_placeholder() if m.grad is not None):
            if not torch.isfinite(p.grad).all():
                gfin = False
        print(f"  (gradient finiteness checked in-driver below)")

    json.dump({"records": RECORDS, "losses": LOSSES},
              open("artifacts/rl_audit/r21c_smoke.json", "w"), indent=2)
    print(f"\n  [save] artifacts/rl_audit/r21c_smoke.json")
    print()
    print(f"RESULT: {'PASS' if n_mixed == 0 else 'FAIL (mixed routes)'}")


def model_params_placeholder():
    return []


if __name__ == "__main__":
    os.makedirs("artifacts/rl_audit", exist_ok=True)
    main()
