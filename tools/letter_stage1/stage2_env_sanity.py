#!/usr/bin/env python3
"""Environment sanity after installing k-means-constrained (numpy 1.x -> 2.2.6).

Checks that the EXISTING MiniOneRec pipeline still imports and that the specific
numpy-2 hazards the LETTER code will hit are understood:
  * torch.from_numpy on a non-writable array
  * torch.from_numpy on a non-contiguous array
"""
import sys

print("=" * 78)
print("ENVIRONMENT SANITY AFTER numpy UPGRADE")
print("=" * 78)
import numpy as np
import torch
print(f"  numpy  = {np.__version__}")
print(f"  torch  = {torch.__version__}")

print("\n--- existing repo modules must still import ---")
sys.path.insert(0, ".")
for m in ("calc", "data", "sasrec"):
    try:
        __import__(m)
        print(f"  OK      import {m}")
    except Exception as e:
        print(f"  FAIL    import {m}: {type(e).__name__}: {str(e)[:120]}")

print("\n--- numpy2 hazard: torch.from_numpy ---")
# writable + contiguous: fine
a = np.ascontiguousarray(np.zeros((3, 4), dtype=np.float32))
try:
    torch.from_numpy(a)
    print("  OK      writable+contiguous")
except Exception as e:
    print(f"  FAIL    writable+contiguous: {type(e).__name__}")

# NON-writable (e.g. np.load mmap, or a read-only view)
b = np.zeros((3, 4), dtype=np.float32)
b.flags.writeable = False
try:
    torch.from_numpy(b)
    print("  OK      read-only")
except Exception as e:
    print(f"  HAZARD  read-only -> {type(e).__name__}: {str(e)[:90]}")

# NON-contiguous (e.g. .T)
c = np.zeros((4, 3), dtype=np.float32).T
try:
    torch.from_numpy(c)
    print("  OK      non-contiguous")
except Exception as e:
    print(f"  HAZARD  non-contiguous -> {type(e).__name__}: {str(e)[:90]}")

# sklearn cluster_centers_ is typically C-contiguous float64
from sklearn.cluster import KMeans
X = np.random.RandomState(0).rand(300, 8).astype(np.float64)
km = KMeans(n_clusters=10, n_init=2, random_state=0).fit(X)
cc = km.cluster_centers_
print(f"\n  sklearn cluster_centers_: dtype={cc.dtype} contig={cc.flags['C_CONTIGUOUS']} "
      f"writeable={cc.flags.writeable}")
try:
    torch.from_numpy(cc)
    print("  OK      torch.from_numpy(cluster_centers_) works")
except Exception as e:
    print(f"  HAZARD  -> {type(e).__name__}: {str(e)[:90]}")

# KMeansConstrained specifically
from k_means_constrained import KMeansConstrained
kmc = KMeansConstrained(n_clusters=10, size_min=5, size_max=60, max_iter=10,
                        n_init=2, n_jobs=2, verbose=False).fit(X)
cc2 = kmc.cluster_centers_
print(f"\n  KMeansConstrained centers: dtype={cc2.dtype} "
      f"contig={cc2.flags['C_CONTIGUOUS']} writeable={cc2.flags.writeable}")
try:
    torch.from_numpy(cc2)
    print("  OK      torch.from_numpy(KMeansConstrained centers) works")
except Exception as e:
    print(f"  HAZARD  -> {type(e).__name__}: {str(e)[:90]}")

print("\n--- the SAFE conversion helper we will use ---")
def safe_tensor(arr):
    import numpy as _np
    a = _np.array(arr, dtype=_np.float32, copy=True)   # always writable+contiguous
    return torch.from_numpy(a)

t = safe_tensor(cc2)
print(f"  OK      safe_tensor -> {tuple(t.shape)} {t.dtype}")
print("\nRESULT: hazard profile captured")
