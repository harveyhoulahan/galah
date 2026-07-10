"""Scaling-law fits over a completed sweep — the analysis behind the paper.

Three artefacts from runs/*/final.json:

  1. IsoFLOP profiles: per budget, val bits/byte vs N with a quadratic fit in
     log N → N_opt(C) (Hoffmann et al. approach 2).
  2. The frontier: power-law fit N_opt = a·C^b (Chinchilla found b≈0.5).
  3. Parametric loss surface L(N, D) = E + A/N^α + B/D^β, fit with Huber loss
     on log-space residuals (approach 3), used for the deployment-constraint
     analysis: minimise L subject to N ≤ N_max(browser budget) rather than
     subject to compute.

  python -m galah.fit --runs runs --figs figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_runs(runs_dir: Path) -> list[dict]:
    out = []
    for f in sorted(runs_dir.glob("*/final.json")):
        out.append(json.loads(f.read_text(encoding="utf-8")))
    if not out:
        raise SystemExit(f"no final.json under {runs_dir}")
    return out


def isoflop_optima(runs: list[dict]) -> list[dict]:
    by_budget: dict[float, list[dict]] = {}
    for r in runs:
        by_budget.setdefault(r["budget_flops"], []).append(r)
    optima = []
    for C, rs in sorted(by_budget.items()):
        rs = sorted(rs, key=lambda r: r["n_params_non_emb"])
        if len(rs) < 3:
            print(f"C={C:.0e}: only {len(rs)} runs, skipping optimum fit")
            continue
        x = np.log([r["n_params_non_emb"] for r in rs])
        y = np.array([r["final_val_bpb"] for r in rs])
        a, b, c = np.polyfit(x, y, 2)
        if a <= 0:
            print(f"C={C:.0e}: profile not convex, taking argmin")
            n_opt = rs[int(np.argmin(y))]["n_params_non_emb"]
            l_opt = float(y.min())
        else:
            n_opt = float(np.exp(-b / (2 * a)))
            l_opt = float(c - b * b / (4 * a))
        optima.append({"C": C, "N_opt": n_opt, "L_opt": l_opt,
                       "D_opt": C / (6 * n_opt), "points": len(rs)})
        print(f"C={C:.0e}: N_opt={n_opt/1e6:.2f}M · D/N={C/(6*n_opt)/n_opt:.0f} · L={l_opt:.4f} bpb")
    return optima


def frontier_fit(optima: list[dict]) -> tuple[float, float]:
    C = np.log([o["C"] for o in optima])
    N = np.log([o["N_opt"] for o in optima])
    b, log_a = np.polyfit(C, N, 1)
    print(f"frontier: N_opt = {np.exp(log_a):.3e} · C^{b:.3f}   (chinchilla: b≈0.50)")
    return float(np.exp(log_a)), float(b)


def parametric_fit(runs: list[dict]) -> dict:
    from scipy.optimize import minimize

    N = np.array([r["n_params_non_emb"] for r in runs], dtype=np.float64)
    D = np.array([r["tokens"] for r in runs], dtype=np.float64)
    L = np.array([r["final_val_bpb"] for r in runs], dtype=np.float64)

    def loss(p):
        e, a, alpha, b, beta = p
        pred = np.exp(e) + np.exp(a) / N**alpha + np.exp(b) / D**beta
        r = np.log(pred) - np.log(L)
        # Huber, delta=1e-3 — robust to the odd diverged run
        d = 1e-3
        return np.sum(np.where(np.abs(r) < d, 0.5 * r**2, d * (np.abs(r) - 0.5 * d)))

    best, best_v = None, np.inf
    for a0 in (2, 6, 10):
        for b0 in (2, 6, 10):
            res = minimize(loss, x0=[np.log(0.8), a0, 0.34, b0, 0.28], method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
            if res.fun < best_v:
                best, best_v = res.x, res.fun
    e, a, alpha, b, beta = best
    fit = {"E": float(np.exp(e)), "A": float(np.exp(a)), "alpha": float(alpha),
           "B": float(np.exp(b)), "beta": float(beta), "huber": float(best_v)}
    print(f"L(N,D) = {fit['E']:.4f} + {fit['A']:.3g}/N^{alpha:.3f} + {fit['B']:.3g}/D^{beta:.3f}")
    return fit


def plots(runs: list[dict], optima: list[dict], figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    by_budget: dict[float, list[dict]] = {}
    for r in runs:
        by_budget.setdefault(r["budget_flops"], []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for C, rs in sorted(by_budget.items()):
        rs = sorted(rs, key=lambda r: r["n_params_non_emb"])
        ax.plot([r["n_params_non_emb"] for r in rs], [r["final_val_bpb"] for r in rs],
                "o-", label=f"C={C:.0e}")
    if optima:
        ax.plot([o["N_opt"] for o in optima], [o["L_opt"] for o in optima],
                "k--x", label="fitted optima")
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("val bits/byte")
    ax.set_title("Galah IsoFLOP profiles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "isoflop.png", dpi=160)

    if len(optima) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.loglog([o["C"] for o in optima], [o["N_opt"] for o in optima], "o-")
        ax.set_xlabel("compute C (FLOPs)")
        ax.set_ylabel("N_opt")
        ax.set_title("compute-optimal frontier")
        fig.tight_layout()
        fig.savefig(figs / "frontier.png", dpi=160)
    print(f"figures → {figs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--figs", type=Path, default=Path("figures"))
    args = ap.parse_args()

    runs = load_runs(args.runs)
    print(f"{len(runs)} completed runs")
    optima = isoflop_optima(runs)
    if len(optima) >= 2:
        frontier_fit(optima)
    fit = parametric_fit(runs) if len(runs) >= 8 else None
    out = {"optima": optima, "parametric": fit}
    (args.runs / "fits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    plots(runs, optima, args.figs)


if __name__ == "__main__":
    main()
