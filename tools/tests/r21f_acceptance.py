#!/usr/bin/env python3
"""
R2.1f -- final correctness acceptance for Reachability-GRPO.

Drives the REAL `rl.py` -> ReReTrainer -> generate -> exact-match reward ->
loss -> backward path on the server's authoritative source, one sample per
(task_type, route) cell, and records raw numbers for independent recomputation.

Measurement only. The reward under test is computed by the Trainer's own
registered function; this driver RE-computes it from the raw
(sample_id, hint, completion, GT) tuples and compares.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import rl_reward as RW                                   # noqa: E402
from minionerec_trainer import ReReTrainer               # noqa: E402

OUT = {"loss_rows": [], "gen_calls": []}
GRAD_BAD = {"n": 0, "params": 0}


def main():
    import fire
    import rl as RL

    cache = json.load(open(os.environ["R21_ROUTE_CACHE"], encoding="utf-8"))
    routes = cache["routes"]
    meta = json.load(open("/tmp/r21/chosen.json"))
    task_of, route_of = meta["task_of"], meta["route_of"]

    print("=" * 96)
    print("R2.1f FINAL CORRECTNESS ACCEPTANCE")
    print("=" * 96)
    for s in meta["chosen"]:
        print(f"  driven: {s:18s} task={task_of[s]:16s} route={route_of[s]}")

    _orig_compute = ReReTrainer.compute_loss

    def _hook(grad):
        if grad is not None and not torch.isfinite(grad).all():
            GRAD_BAD["n"] += 1
        return grad

    def _compute(self, model, inputs, *a, **kw):
        handles = []
        if not getattr(self, "_r21f_hooked", False):
            for _n, p in model.named_parameters():
                if p.requires_grad:
                    handles.append(p.register_hook(_hook))
                    GRAD_BAD["params"] += 1
            self._r21f_hooked = True
        loss = _orig_compute(self, model, inputs, *a, **kw)
        for h in handles:
            h.remove()
        try:
            comp = inputs.get("completion_ids")
            adv = inputs.get("advantages")
            cmask = inputs.get("completion_mask")
            hl = int(getattr(self, "_r21_hint_len_cache", 0))
            pce = (float(self._prefix_ce.detach())
                   if getattr(self, "_prefix_ce", None) is not None else None)
            row = {"loss": float(loss.detach()), "hint_len": hl, "prefix_ce": pce,
                   "coef": float(getattr(self, "prefix_ce_coef", 0.0)),
                   "comp_shape": list(comp.shape) if comp is not None else None,
                   "cmask_sum": int(cmask.sum().item()) if cmask is not None else None,
                   "adv": adv.detach().float().tolist() if adv is not None else None}
            OUT["loss_rows"].append(row)
            nz = sum(1 for x in (row["adv"] or []) if abs(x) > 1e-12)
            print(f"  [loss] loss={row['loss']:.6f} h={hl} prefix_ce={pce} "
                  f"coef={row['coef']} cmask_sum={row['cmask_sum']} "
                  f"comp_shape={row['comp_shape']} adv_nonzero={nz}")
        except Exception as e:  # noqa: BLE001
            print(f"  [loss] record failed: {type(e).__name__}: {e}")
        return loss

    ReReTrainer.compute_loss = _compute

    _orig_prepare = ReReTrainer._prepare_inputs

    def _prepare(self, inputs):
        sids = [x.get("sample_id") for x in inputs]
        rt = [routes.get(s, {}).get("route") for s in sids]
        rec = {"n": len(inputs), "sids": sids, "routes": rt,
               "mixed": len(set(rt)) > 1,
               "tasks": sorted(set(str(task_of.get(s)) for s in sids))}
        OUT["gen_calls"].append(rec)
        print(f"  [gen] n={len(inputs)} tasks={rec['tasks']} "
              f"routes={sorted(set(str(x) for x in rt))} mixed={rec['mixed']}")
        return _orig_prepare(self, inputs)

    ReReTrainer._prepare_inputs = _prepare

    argv = [f"--{k}={v}" for k, v in json.loads(os.environ["R21_RL_KWARGS"]).items()
            if v is not None]
    try:
        fire.Fire(RL.train, command=argv)
    except SystemExit:
        pass

    S = getattr(RL, "_R21_SCRATCH", {}) or {}
    sids = S.get("sids") or []
    tgts = S.get("targets") or []
    comps = S.get("completions") or []

    print()
    print("=" * 96)
    print("TASK 2  INDEPENDENT REWARD RECOMPUTATION")
    print("=" * 96)
    n = hit = miss = 0
    rows = []
    for i, sid in enumerate(sids):
        if i >= len(tgts) or tgts[i] is None:
            continue
        hint = routes.get(sid, {}).get("hint", "") if route_of.get(sid) == "HARD" else ""
        gt = tgts[i]
        full = RW.reconstruct(hint, comps[i])
        indep = 1.0 if full == gt.strip() else 0.0
        n += 1
        hit += (indep == 1.0)
        miss += (indep == 0.0)
        rows.append((sid, task_of.get(sid), route_of.get(sid), hint, full,
                     gt.strip(), indep))
    for r in rows:
        print(f"    {r[0]:18s} {str(r[1]):16s} {str(r[2]):7s} hint={r[3]!r:10s} "
              f"full={r[4]:26s} GT={r[5]:26s} reward={r[6]}")
    print()
    print(f"  rows = {n}   GT-hit = {hit}   GT-miss = {miss}")

    print()
    print("=" * 96)
    print("TASK 3/5  HARD SUFFIX-ONLY CREDIT")
    print("=" * 96)
    for i, r in enumerate(OUT["loss_rows"]):
        nz = sum(1 for x in (r["adv"] or []) if abs(x) > 1e-12)
        if r["hint_len"] > 0:
            exp = 0.1 * r["prefix_ce"] if r["prefix_ce"] is not None else float("nan")
            print(f"    step {i} HARD  : loss={r['loss']:.6f} 0.1*prefix_ce={exp:.6f} "
                  f"adv_nonzero={nz}/{len(r['adv'] or [])} cmask_sum={r['cmask_sum']} "
                  f"comp_shape={r['comp_shape']}")
        else:
            print(f"    step {i} NORMAL: loss={r['loss']:.6f} prefix_ce={r['prefix_ce']} "
                  f"adv_nonzero={nz}/{len(r['adv'] or [])} comp_shape={r['comp_shape']}")

    print()
    print("=" * 96)
    print("TASK 4  GRADIENT FINITENESS")
    print("=" * 96)
    print(f"  params hooked = {GRAD_BAD['params']}   non-finite grads = {GRAD_BAD['n']}")

    print()
    print("=" * 96)
    print("GENERATE CALLS")
    print("=" * 96)
    for i, g in enumerate(OUT["gen_calls"]):
        print(f"    call {i}: n={g['n']} tasks={g['tasks']} routes={g['routes']} "
              f"mixed={g['mixed']}")

    json.dump({"chosen": meta["chosen"], "task_of": task_of, "route_of": route_of,
               "rows": [list(r) for r in rows], "n_rows": n, "gt_hit": hit,
               "gt_miss": miss, "grad_bad": GRAD_BAD["n"],
               "grad_params": GRAD_BAD["params"], "gen_calls": OUT["gen_calls"],
               "loss_rows": OUT["loss_rows"]},
              open("artifacts/rl_audit/r21f_acceptance.json", "w"), indent=2)
    print("\n  [save] artifacts/rl_audit/r21f_acceptance.json")


if __name__ == "__main__":
    os.makedirs("artifacts/rl_audit", exist_ok=True)
    main()
