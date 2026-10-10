#!/usr/bin/env python3
"""
Stage 4 / step 2 -- prove the two regularisation terms are EXCLUDED from baseline
B's optimisation objective, by PERTURBATION (not by re-deriving their value).

Why perturbation instead of the coefficient identity
----------------------------------------------------
The identity `total(a,b) - total(0,0) == a*CF + b*diversity` holds exactly for the
CF term (verified separately: rel_err 2.7e-14 in float64). It CANNOT be checked for
the diversity term by recomputation, because `vq.diversity_loss` samples its
positive target with `random.choice` on every forward -- the term is stochastic, so
any "measure it again" comparison is confounded by a different draw.

Perturbation avoids that and tests exactly what matters:

  alpha = 0  =>  d(total)/d(theta) must not depend on the collaborative signal.
                 Verified by replacing cf_embedding with a DIFFERENT random matrix
                 and requiring every parameter gradient to be bitwise identical.

  beta  = 0  =>  d(total)/d(theta) must not depend on the diversity term.
                 Verified by permuting the cluster labels diversity_loss consumes
                 and requiring every parameter gradient to be bitwise identical.
                 (The stochastic draw changes, but beta multiplies it by zero, so
                 nothing may reach the gradient.)

Scale controls (alpha=100 / beta=100) must make the same perturbations change the
gradients, which proves the test is not vacuous.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LETTER = os.path.abspath(os.path.join(HERE, "..", "..", "rq", "letter"))
sys.path.insert(0, LETTER)
from models.rqvae import RQVAE  # noqa: E402

S1 = "artifacts/letter_stage1"
OUT = "artifacts/letter_stage4_content_only"
ARCH = dict(in_dim=896, num_emb_list=[256, 256, 256, 256], e_dim=32,
            layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
            loss_type="mse", quant_loss_weight=1.0, kmeans_init=True,
            kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0, 0.003], sk_iters=50,
            n_clusters=10, sample_strategy="all")


def compute_labels(model, n_clusters=10):
    from k_means_constrained import KMeansConstrained
    labels = {}
    for i, vq in enumerate(model.rq.vq_layers):
        W = vq.embedding.weight.detach().cpu().numpy().astype(np.float64)
        n = W.shape[0]
        size_min = min(n // (n_clusters * 2), 10)
        size_max = max(n_clusters * 6, size_min * 4)
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min,
                                size_max=size_max, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False).fit(W)
        labels[str(i)] = clf.labels_.tolist()
    return labels


def grads(alpha, beta, cf, state, X, seen, labels, seed=42, dtype=torch.float64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = RQVAE(alpha=alpha, beta=beta, cf_embedding=cf, **ARCH)
    m.load_state_dict(state, strict=True)
    m = m.to(dtype).train()
    d = torch.from_numpy(X[seen]).to(dtype)
    idx = torch.from_numpy(seen.astype(np.int64))
    out, rq_loss, _, x_q = m(d, labels, use_sk=True)
    total, cf_loss, recon, quant = m.compute_loss(out, rq_loss, idx, x_q, xs=d)
    m.zero_grad()
    total.backward()
    g = {n: (p.grad.detach().clone() if p.grad is not None else None)
         for n, p in m.named_parameters()}
    return float(total.detach()), float(cf_loss.detach()), g


def max_diff(g1, g2):
    mx, nd = 0.0, 0
    for n in sorted(set(g1) | set(g2)):
        a, b = g1.get(n), g2.get(n)
        if a is None or b is None:
            continue
        d = (a - b).abs().max().item()
        mx = max(mx, d)
        if d != 0.0:
            nd += 1
    return nd, mx


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 84)
    print("STEP 2 -- PERTURBATION PROOF OF REGULARISATION EXCLUSION")
    print("=" * 84)

    X = np.load(f"{S1}/text_qwen05b.npy").astype(np.float32)
    CF = torch.load(f"{S1}/sasrec32_item_emb.pt", weights_only=True).float().numpy()
    mask = np.load(f"{S1}/cf_seen_mask.npy")
    seen = np.where(mask)[0][:1024]
    print(f"  X {X.shape}  CF {CF.shape}  batch {seen.shape}")

    torch.manual_seed(1234)
    ref = RQVAE(alpha=0.0, beta=0.0, cf_embedding=CF, **ARCH)
    state = {k: v.clone() for k, v in ref.state_dict().items()}

    CF64 = np.asarray(CF, dtype=np.float64)
    rng = np.random.RandomState(999)
    CF_other = (rng.randn(*CF64.shape) * CF64.std()).astype(np.float64)
    rel = float(np.linalg.norm(CF_other - CF64) / np.linalg.norm(CF64))
    print(f"  CF perturbation  : independent N(0,{CF64.std():.4f}) matrix, "
          f"relative change = {rel:.4f}")

    LBL = compute_labels(ref)
    rng2 = np.random.RandomState(555)
    LBL_perm = {k: rng2.permutation(np.array(v)).tolist() for k, v in LBL.items()}
    n_moved = sum(1 for k in LBL for a, b in zip(LBL[k], LBL_perm[k]) if a != b)
    print(f"  label perturbation: {n_moved}/1024 code->cluster assignments moved")

    fails = []

    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            fails.append(label)
        return ok

    # ---------------- alpha --------------------------------------------------
    print()
    print("  --- alpha = 0: gradients must be independent of cf_embedding ---")
    for a in (0.0, 0.01, 100.0):
        tA, cfA, gA = grads(a, 0.0, CF64, state, X, seen, LBL)
        tB, cfB, gB = grads(a, 0.0, CF_other, state, X, seen, LBL)
        nd, mx = max_diff(gA, gB)
        print(f"    alpha={a:<7} cf_A={cfA:>10.6f} cf_B={cfB:>10.6f}  "
              f"grad n_diff={nd:>4d} max={mx:.3e}")
        if a == 0.0:
            chk(nd == 0 and mx == 0.0,
                "alpha=0: ALL gradients BITWISE identical under CF perturbation",
                f"n_diff={nd} max={mx:.3e}")
        else:
            chk(nd > 0,
                f"SCALE CONTROL alpha={a}: CF perturbation DOES change gradients",
                f"{nd} params differ, max {mx:.3e}")

    # ---------------- beta ---------------------------------------------------
    print()
    print("  --- beta = 0: gradients must be independent of the cluster labels ---")
    for b in (0.0, 0.0001, 100.0):
        tA, _, gA = grads(0.0, b, CF64, state, X, seen, LBL)
        tB, _, gB = grads(0.0, b, CF64, state, X, seen, LBL_perm)
        nd, mx = max_diff(gA, gB)
        print(f"    beta={b:<8} grad n_diff={nd:>4d} max={mx:.3e}")
        if b == 0.0:
            chk(nd == 0 and mx == 0.0,
                "beta=0: ALL gradients BITWISE identical under label perturbation",
                f"n_diff={nd} max={mx:.3e}")
        else:
            chk(nd > 0,
                f"SCALE CONTROL beta={b}: label perturbation DOES change gradients",
                f"{nd} params differ, max {mx:.3e}")

    # ---------------- effective values --------------------------------------
    print()
    print("  --- effective configuration on the B model ---")
    mB = RQVAE(alpha=0.0, beta=0.0, cf_embedding=CF, **ARCH)
    print(f"    effective_alpha      = {mB.alpha}")
    print(f"    effective_beta       = {mB.beta}")
    print(f"    per-level vq[i].beta = {[vq.beta for vq in mB.rq.vq_layers]}")
    print(f"    per-level vq[i].mu   = {[vq.mu for vq in mB.rq.vq_layers]}")
    print(f"    quant_loss_weight    = {mB.quant_loss_weight}")
    print(f"    sk_epsilons          = {[vq.sk_epsilon for vq in mB.rq.vq_layers]}")
    chk(mB.alpha == 0.0, "effective_alpha == 0")
    chk(all(vq.beta == 0.0 for vq in mB.rq.vq_layers), "effective_beta == 0")

    print()
    print(f"  failures = {fails}")
    print(f"  RESULT: {'ALL PASS' if not fails else 'FAIL'}")
    json.dump({"cf_perturbation_relative": rel,
               "label_assignments_moved": n_moved,
               "effective_alpha": 0.0, "effective_beta": 0.0,
               "failures": fails},
              open(f"{OUT}/step2_grad_proof.json", "w"), indent=2)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
