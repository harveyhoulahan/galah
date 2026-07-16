"""IsoFLOP sweep runner.

For each compute budget C, train every ladder rung whose expected D/N ratio
is sane (a rung is skipped when the budget would give it fewer than 2 or more
than 2000 bytes per parameter — far outside any plausible optimum, wasted
compute). Runs execute sequentially on one GPU; each writes final.json, and
the sweep resumes for free by skipping runs whose final.json already exists.

  python -m galah.sweep --budgets 1e15,3e15,1e16,3e16,1e17,3e17 --data data/

galah.fit turns the resulting runs/ directory into the paper's curves.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .model import LADDER

try:
    from setproctitle import setproctitle
except ImportError:
    def setproctitle(_: str) -> None:  # shared boxes without the package: no-op
        pass


def main() -> None:
    setproctitle("train-worker-sweep")  # shared box: keep argv off other users' ps/btop
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1e15,3e15,1e16,3e16,1e17")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tokens-per-step", type=int, default=131072)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    plan: list[tuple[str, float, float]] = []
    for C in budgets:
        for rung, cfg in LADDER.items():
            n = cfg.n_params_non_emb
            d_over_n = C / (cfg.flops_per_token() * 1.0) / n  # ≈ tokens/param
            if 2 <= d_over_n <= 2000:
                plan.append((rung, C, d_over_n))

    print(f"sweep: {len(plan)} runs over {len(budgets)} budgets")
    for rung, C, dn in plan:
        print(f"  C={C:.0e}  {rung:<12} D/N≈{dn:>7.1f}")
    if args.dry:
        return

    for i, (rung, C, _) in enumerate(plan):
        name = f"{rung}_C{C:.0e}".replace("+", "")
        if (Path(args.out) / name / "final.json").exists():
            print(f"[{i+1}/{len(plan)}] {name} — done, skipping")
            continue
        print(f"[{i+1}/{len(plan)}] {name}")
        cmd = [sys.executable, "-m", "galah.train", "--rung", rung, "--budget", f"{C:.3e}",
               "--data", args.data, "--out", args.out, "--tokens-per-step", str(args.tokens_per_step)]
        if args.compile:
            cmd.append("--compile")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"run {name} failed ({r.returncode}); fix and re-launch to resume")

    manifest = {"budgets": budgets, "runs": [p[0] + f"_C{p[1]:.0e}".replace("+", "") for p in plan]}
    Path(args.out, "sweep.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("sweep complete.")


if __name__ == "__main__":
    main()
