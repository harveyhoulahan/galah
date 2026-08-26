"""Re-run the frontier's load-bearing configs at extra seeds.

The power-law frontier N_opt = a·C^b is fitted only on *interior* (uncensored)
isoFLOP optima. Those N_opt values — and the local slopes between consecutive
ones — are set by the observed minimum of each interior budget's profile.
Censored edge-minima (the 1e15 rung, whose vertex sits on the ladder edge)
never enter the fit, so they are not repeated.

This script discovers those (rung, C) pairs from a completed sweep and
launches them again through the frozen galah.train recipe at extra seeds,
suffixing the run name (`-seed2337`, …). Finished runs (final.json present)
are skipped, same as the sweep.

  python -m galah.seed_repeats --runs runs --out runs --compile
  python -m galah.seed_repeats --dry
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .fit import load_bearing_configs, load_runs, partition_runs

try:
    from setproctitle import setproctitle
except ImportError:
    def setproctitle(_: str) -> None:
        pass


# Published interior discrete minima from the completed lychee sweep, used
# only when --runs is empty so the plan is inspectable off-box. Recomputed
# from runs/ whenever final.json files are present.
FALLBACK_OPTIMA: list[tuple[str, float]] = [
    ("galah-0.3m", 3e15),   # interior; local slope into 1e16
    ("galah-1.5m", 1e16),   # local b: 0.92 →
    ("galah-5.5m", 3e16),   #           0.87 →
    ("galah-18m",  1e17),   #           0.74 →
    ("galah-38m",  3e17),   #           0.61
    ("galah-69m",  1e18),
]

DEFAULT_SEEDS = (2337, 3337, 4337)


def _plan(runs_dir: Path) -> list[tuple[str, float]]:
    if runs_dir.exists() and any(runs_dir.glob("*/final.json")):
        runs = load_runs(runs_dir)
        main, _ = partition_runs(runs)
        configs = load_bearing_configs(main)
        if configs:
            return configs
        print(f"no interior optima in {runs_dir}; falling back to published minima")
    else:
        print(f"no sweep under {runs_dir}; using published interior minima")
    return list(FALLBACK_OPTIMA)


def _run_name(rung: str, budget: float, seed: int) -> str:
    return f"{rung}_C{budget:.0e}".replace("+", "") + f"-seed{seed}"


def main() -> None:
    setproctitle("train-worker-seeds")
    ap = argparse.ArgumentParser(
        description="Re-run interior isoFLOP optima at extra seeds (frozen recipe).")
    ap.add_argument("--runs", type=Path, default=Path("runs"),
                    help="completed sweep; used to discover the load-bearing configs")
    ap.add_argument("--out", type=Path, default=Path("runs"),
                    help="where seed-repeat runs write (train --out)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="comma-separated extra seeds")
    ap.add_argument("--tokens-per-step", type=int, default=131072)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    configs = _plan(args.runs)

    print(f"seed-repeats: {len(configs)} interior optima × {len(seeds)} seeds")
    for rung, C in configs:
        print(f"  C={C:.0e}  {rung}")
    if args.dry:
        for rung, C in configs:
            for seed in seeds:
                print(f"    would run {_run_name(rung, C, seed)}")
        return

    jobs = [(rung, C, seed) for rung, C in configs for seed in seeds]
    for i, (rung, C, seed) in enumerate(jobs):
        name = _run_name(rung, C, seed)
        dest = args.out / name / "final.json"
        if dest.exists():
            print(f"[{i+1}/{len(jobs)}] {name} — done, skipping")
            continue
        print(f"[{i+1}/{len(jobs)}] {name}")
        cmd = [
            sys.executable, "-m", "galah.train",
            "--rung", rung, "--budget", f"{C:.3e}",
            "--seed", str(seed), f"--suffix=-seed{seed}",
            "--data", args.data, "--out", str(args.out),
            "--tokens-per-step", str(args.tokens_per_step),
        ]
        if args.compile:
            cmd.append("--compile")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"run {name} failed ({r.returncode}); fix and re-launch to resume")
    print("seed-repeats complete.")


if __name__ == "__main__":
    main()
