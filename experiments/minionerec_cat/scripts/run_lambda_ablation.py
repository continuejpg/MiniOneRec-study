"""Phase 2B driver -- run the two lambda ablations sequentially, after B2-seed7.

One GPU, so the runs must serialise.  Waiting on a sentinel keeps this simple and
avoids two training processes contending for VRAM.

Each arm: train -> extract_index -> rebuild_derived -> verify_chain -> audit_sid,
all under its own run tag so nothing overwrites B0 or the existing B2 (seed 2024).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"D:\IT\CODE\PYTHON\project1\MiniOneRec")
PY = sys.executable
RUNS = REPO / "experiments" / "minionerec_cat" / "runs"
CF = "experiments/minionerec_cat/runs/b2/collab/cf_embedding_b2.npy"
CFM = "experiments/minionerec_cat/runs/b2/collab/cf_mask_b2.npy"

ARMS = [
    ("b2_lam005", "0.05"),
    ("b2_lam010", "0.10"),
]


def run(cmd: list[str], log: Path) -> int:
    """Run a stage, capturing its output to ``log``.

    Output is redirected to a FILE, never piped through a consumer that might close
    early: a truncated reader (e.g. ``Select-Object -Last N`` in PowerShell) closes
    the pipe and kills the child mid-write, which silently loses whatever the child
    writes last -- including its JSON report.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f"\nexit={p.returncode}\n")
    return p.returncode


def _chain_done(tag: str) -> bool:
    """True when this tag's full chain has produced its verification report."""
    return (RUNS / tag / "reports" / f"{tag}_chain_verification.json").exists()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wait-for", default=None,
                    help="tag whose chain must finish first (optional)")
    ap.add_argument("--arms", nargs="+", default=[a[0] for a in ARMS])
    args = ap.parse_args(argv)

    if args.wait_for:
        print(f"waiting for {args.wait_for} chain ...", flush=True)
        waited = 0
        while not _chain_done(args.wait_for):
            time.sleep(20)
            waited += 20
            if waited > 5400:
                print(f"  giving up after {waited//60} min", flush=True)
                return 2
        print(f"  {args.wait_for} ready after {waited//60} min", flush=True)

    todo = [(t, l) for t, l in ARMS if t in args.arms]
    for tag, lam in todo:
        log = RUNS / tag / "logs" / "driver.log"
        print(f"\n=== {tag}  lambda_cf={lam} ===", flush=True)

        rc = run([PY, "-m", "experiments.minionerec_cat.src.train_rqvae",
                  "--tag", tag, "--seed", "2024", "--num-workers", "0",
                  "--device", "cuda:0", "--lambda-cf", lam,
                  "--cf-embedding", CF, "--cf-mask", CFM, "--cf-temperature", "1.0"],
                 log)
        print(f"  train rc={rc}", flush=True)
        if rc != 0:
            print(f"  STOP: training failed for {tag}", flush=True)
            continue

        for stage, extra in [
            ("extract_index", ["--seed", "2024", "--device", "cuda:0", "--batch-size", "4096"]),
            ("rebuild_derived", []),
            ("verify_chain", []),
            ("audit_sid", ["--coherence-sample", "800"]),
        ]:
            rc = run([PY, "-m", f"experiments.minionerec_cat.src.{stage}",
                      "--tag", tag, *extra],
                     RUNS / tag / "logs" / f"{stage}.log")
            print(f"  {stage:16} rc={rc}", flush=True)
            if rc != 0:
                print(f"  STOP: {stage} failed for {tag}", flush=True)
                break
    print("\ndriver complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
