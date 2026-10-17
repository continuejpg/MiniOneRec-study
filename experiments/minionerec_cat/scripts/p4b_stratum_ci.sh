#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/bin:/bin
/root/miniconda3/bin/python - <<'PYEOF'
import json, math, pathlib
import numpy as np
D = pathlib.Path("/root/autodl-tmp")
OUT = D/"p4b_analysis"
STRATA = ("freq0","low(1-2)","mid(3-20)",">high".replace(">","")+"" )
STRATA = ("freq0","low(1-2)","mid(3-20)","high(>20)")
def gain(r): return (1.0/math.log(r+2))/(1.0/math.log(2))
c0 = json.loads((OUT/"C0_valid_per_row.json").read_text())
c1 = json.loads((OUT/"C1_valid_per_row.json").read_text())
users = [r["user_id"] for r in c0]

def boot(a, b, cl=None, nb=10000, seed=2024):
    a=np.asarray(a,float); b=np.asarray(b,float); d=b-a; n=len(d)
    rng=np.random.default_rng(seed)
    if cl is None:
        s=d[rng.integers(0,n,size=(nb,n))].mean(axis=1)
    else:
        cl=np.asarray(cl); u,inv=np.unique(cl,return_inverse=True)
        o=np.argsort(inv,kind="stable"); si=inv[o]
        mem=[o[np.searchsorted(si,i,"left"):np.searchsorted(si,i,"right")] for i in range(len(u))]
        s=np.empty(nb)
        for t in range(nb):
            p=rng.integers(0,len(u),size=len(u)); s[t]=d[np.concatenate([mem[x] for x in p])].mean()
    lo,hi=float(np.quantile(s,.025)),float(np.quantile(s,.975))
    return 100*float(d.mean()), 100*lo, 100*hi, bool(lo>0 or hi<0)

print("="*100)
print("PER-STRATUM PAIRED BOOTSTRAP, C1 - C0 (VALID, 4,532 rows, same rows/order)")
print("="*100)
print(f"  {'stratum':<12}{'rows':>6}{'users':>7}{'HR d(pp)':>11}{'95% CI':>24}{'excl0':>7}"
      f"{'NDCG d(pp)':>12}{'95% CI':>24}{'excl0':>7}")
summ={}
for k in STRATA:
    ix=[i for i,r in enumerate(c0) if r["stratum"]==k]
    ha=[1 if c0[i]["repo_rank"] is not None else 0 for i in ix]
    hb=[1 if c1[i]["repo_rank"] is not None else 0 for i in ix]
    na=[gain(c0[i]["repo_rank"]) if c0[i]["repo_rank"] is not None else 0.0 for i in ix]
    nb_=[gain(c1[i]["repo_rank"]) if c1[i]["repo_rank"] is not None else 0.0 for i in ix]
    uu=[users[i] for i in ix]
    dh, lh, hh, eh = boot(ha, hb, uu)
    dn, ln, hn, en = boot(na, nb_, uu)
    summ[k]={"rows":len(ix),"users":len(set(uu)),
             "HR_delta_pp":round(dh,4),"HR_ci":[round(lh,4),round(hh,4)],"HR_excl0":eh,
             "NDCG_delta_pp":round(dn,4),"NDCG_ci":[round(ln,4),round(hn,4)],"NDCG_excl0":en}
    print(f"  {k:<12}{len(ix):>6}{len(set(uu)):>7}{dh:>+11.4f}   [{lh:>+7.4f},{hh:>+7.4f}]"
          f"{str(eh):>7}{dn:>+12.4f}   [{ln:>+7.4f},{hn:>+7.4f}]{str(en):>7}")

print()
print("  reading: positive HR delta on freq0/low/mid = the tail improved;")
print("           large negative on high = the head lost accuracy.")
(OUT/"p4b_stratum_bootstrap.json").write_text(json.dumps(summ,indent=1))
print(f"\n  written {OUT/'p4b_stratum_bootstrap.json'}")
PYEOF
