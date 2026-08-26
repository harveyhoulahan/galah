"""Scaling-law fits over a completed sweep — the analysis behind the paper.

Artefacts from runs/*/final.json (+ log.jsonl):

  1. Divergence screen: a healthy cosine-to-floor run ends at its lowest
     smoothed train loss; a run whose final EMA sits >15% above its own best
     hit a mid-schedule loss spike and never recovered. Those runs are
     excluded from every fit (as in Hoffmann et al.) but kept in fits.json
     and plotted — the stability envelope of the frozen recipe is itself a
     result the paper reports.
  2. IsoFLOP profiles: per budget, val bits/byte vs N with a quadratic fit in
     log N → N_opt(C) (Hoffmann et al. approach 2). A budget whose observed
     minimum sits on the edge of the ladder is CENSORED: its true optimum
     lies at/beyond the sampled range, so it enters the frontier only as an
     upper bound, never as a fit point.
  3. The frontier: power-law fit N_opt = a·C^b over interior (uncensored)
     budgets. Chinchilla found b≈0.50 at BPE scale; Kaplan-regime small-scale
     studies find much steeper b — see Porian et al. 2024, Pearce & Song 2024.
  4. Parametric loss surface L(N, D) = E + A/N^α + B/D^β, Huber in log space
     (approach 3), used for the deployment-constraint analysis: minimise L
     subject to N ≤ N_max(browser budget) rather than subject to compute.
     Consistency check printed: b implied by the surface is β/(α+β).

D accounting: tokens were allocated as C / flops_per_token(N) — which
includes the attention quadratic (64% of FLOPs at the 0.1M rung, 18% at
113M) — so D_opt is reported the same way, NOT as C/(6·N_opt).

Optional runs_stab/ (stability-annex reruns of diverged configs at reduced
lr) is overlaid on the isoflop figure as hollow markers; it never enters
the frozen-recipe point estimates. Seed-repeats of the interior discrete
minima (`-seedNNNN` suffix, from galah.seed_repeats) are also kept out of
the isoFLOP / parametric fits; their across-seed standard deviation is
attached to each interior optimum and used as error bars on the frontier.

  python -m galah.fit --runs runs --stab runs_stab --figs figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DIVERGENCE_FACTOR = 1.15  # final EMA > 1.15 × best EMA ⇒ post-spike, excluded
MAIN_NAME = re.compile(r"^galah-[\d.]+m_C\d+e\d+$")
SEED_SUFFIX = re.compile(r"-seed(\d+)$")


def load_runs(runs_dir: Path) -> list[dict]:
    out = []
    for f in sorted(runs_dir.glob("*/final.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        log_path = f.parent / "log.jsonl"
        r["diverged"] = False
        if log_path.exists():
            emas = [json.loads(l)["ema"] for l in log_path.read_text(encoding="utf-8").splitlines() if l]
            if emas:
                r["min_ema"] = min(emas)
                r["diverged"] = emas[-1] > DIVERGENCE_FACTOR * min(emas)
        out.append(r)
    if not out:
        raise SystemExit(f"no final.json under {runs_dir}")
    return out


def partition_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split unsuffixed sweep runs from `-seedNNNN` repeats (and other suffixes)."""
    main, repeats = [], []
    for r in runs:
        name = r.get("name", "")
        if SEED_SUFFIX.search(name):
            repeats.append(r)
        elif MAIN_NAME.match(name) and r.get("lr_scale", 1.0) == 1.0:
            main.append(r)
        else:
            repeats.append(r)
    return main, repeats


def _vertex(rs: list[dict], y: np.ndarray | None = None) -> dict:
    """Quadratic-in-logN vertex for one budget. y overrides val bpb (seed sweeps)."""
    rs = sorted(rs, key=lambda r: r["n_params_non_emb"])
    x = np.log([r["n_params_non_emb"] for r in rs])
    if y is None:
        y = np.array([r["final_val_bpb"] for r in rs], dtype=np.float64)
    imin = int(np.argmin(y))
    censored = imin in (0, len(rs) - 1)
    a, b, c = np.polyfit(x, y, 2)
    # A vertex outside the sampled range is not a located optimum: clipping it to
    # the edge would pair that N with an L the parabola predicts somewhere else
    # (it can even go negative). Treat it as censored, like an edge minimum.
    if a <= 0 or censored or not x[0] <= -b / (2 * a) <= x[-1]:
        n_opt = float(rs[imin]["n_params_non_emb"])
        l_opt = float(y.min())
        censored = True
    else:
        n_opt = float(np.exp(-b / (2 * a)))
        l_opt = float(c - b * b / (4 * a))
    fpt_opt = float(np.exp(np.interp(np.log(n_opt), x,
                                     np.log([r["flops_per_token"] for r in rs]))))
    return {
        "N_opt": n_opt, "L_opt": l_opt, "censored": censored,
        "fpt_opt": fpt_opt, "imin": imin, "rung": rs[imin]["rung"],
        "name": rs[imin]["name"],
    }


def isoflop_optima(runs: list[dict], verbose: bool = True) -> list[dict]:
    """Quadratic-in-logN vertex per budget, on clean main-sweep runs only."""
    by_budget: dict[float, list[dict]] = {}
    for r in runs:
        if not r["diverged"]:
            by_budget.setdefault(r["budget_flops"], []).append(r)
    optima = []
    for C, rs in sorted(by_budget.items()):
        rs = sorted(rs, key=lambda r: r["n_params_non_emb"])
        if len(rs) < 3:
            if verbose:
                print(f"C={C:.0e}: only {len(rs)} clean runs, skipping optimum fit")
            continue
        v = _vertex(rs)
        optima.append({
            "C": C, "N_opt": v["N_opt"], "L_opt": v["L_opt"],
            "D_opt": C / v["fpt_opt"], "points": len(rs),
            "censored": v["censored"], "rung": v["rung"], "name": v["name"],
        })
        if verbose:
            tag = "  [CENSORED — edge minimum, upper bound only]" if v["censored"] else ""
            print(f"C={C:.0e}: N_opt={v['N_opt']/1e6:.2f}M · D_opt={C/v['fpt_opt']/1e9:.2f}GB "
                  f"· D/N={C/v['fpt_opt']/v['N_opt']:.0f} · L={v['L_opt']:.4f} bpb"
                  f" · min {v['rung']}{tag}")
    return optima


def load_bearing_configs(runs: list[dict]) -> list[tuple[str, float]]:
    """Discrete observed minima of interior (uncensored) budgets.

    These are the only configs the frontier power law — and the local slopes
    between consecutive N_opt — actually depend on. Censored edge budgets
    are dropped.
    """
    return [(o["rung"], o["C"]) for o in isoflop_optima(runs, verbose=False)
            if not o["censored"]]


def attach_seed_errors(optima: list[dict], main: list[dict],
                       repeats: list[dict]) -> None:
    """Across-seed std of L at each interior discrete min, propagated to N_opt.

    Seed repeats of the winning rung are substituted into that budget's
    isoFLOP profile and the quadratic is refit; the spread of the resulting
    vertices is N_opt_std. Mutates optima in place.
    """
    by_budget: dict[float, list[dict]] = {}
    for r in main:
        if not r["diverged"]:
            by_budget.setdefault(r["budget_flops"], []).append(r)
    pool = [r for r in main + repeats
            if r.get("lr_scale", 1.0) == 1.0 and not r.get("diverged")]

    for o in optima:
        if o["censored"]:
            continue
        rs = sorted(by_budget.get(o["C"], []), key=lambda r: r["n_params_non_emb"])
        if len(rs) < 3:
            continue
        imin = next(i for i, r in enumerate(rs) if r["rung"] == o["rung"])
        vals = [r["final_val_bpb"] for r in pool
                if r["rung"] == o["rung"] and r["budget_flops"] == o["C"]]
        o["n_seeds"] = len(vals)
        o["L_seeds"] = [float(v) for v in vals]
        if len(vals) < 2:
            continue
        o["L_std"] = float(np.std(vals, ddof=1))
        n_opts = []
        y0 = np.array([r["final_val_bpb"] for r in rs], dtype=np.float64)
        for v in vals:
            y = y0.copy()
            y[imin] = v
            n_opts.append(_vertex(rs, y)["N_opt"])
        o["N_opt_std"] = float(np.std(n_opts, ddof=1))
        print(f"C={o['C']:.0e}  {o['rung']}: {len(vals)} seeds · "
              f"L={np.mean(vals):.4f}±{o['L_std']:.4f} bpb · "
              f"N_opt_std={o['N_opt_std']/1e6:.3f}M")


def frontier_fit(optima: list[dict]) -> dict | None:
    interior = [o for o in optima if not o["censored"]]
    if len(interior) < 2:
        print("frontier: <2 interior optima, no fit")
        return None
    C = np.log([o["C"] for o in interior])
    N = np.log([o["N_opt"] for o in interior])
    stds = [o.get("N_opt_std") for o in interior]
    weighted = all(s is not None and s > 0 for s in stds)
    kwargs: dict = {}
    if weighted:
        # δ log N ≈ σ_N / N; polyfit weights are 1/σ on the y-coordinate.
        kwargs["w"] = np.array([o["N_opt"] / o["N_opt_std"] for o in interior])
        kwargs["cov"] = True
        (b, log_a), cov = np.polyfit(C, N, 1, **kwargs)
        b_std = float(np.sqrt(cov[0, 0]))
        extra = f"  (WLS · σ_b={b_std:.3f})"
    else:
        b, log_a = np.polyfit(C, N, 1)
        b_std = None
        extra = ""
    print(f"frontier: N_opt = {np.exp(log_a):.3e} · C^{b:.3f} over {len(interior)} "
          f"interior budgets  (Chinchilla b≈0.50; Kaplan-regime small-scale runs steeper)"
          f"{extra}")
    out = {"a": float(np.exp(log_a)), "b": float(b), "n_budgets": len(interior),
           "weighted": weighted}
    if b_std is not None:
        out["b_std"] = b_std
    return out


def parametric_fit(runs: list[dict]) -> dict:
    from scipy.optimize import minimize

    clean = [r for r in runs if not r["diverged"]]
    N = np.array([r["n_params_non_emb"] for r in clean], dtype=np.float64)
    D = np.array([r["tokens"] for r in clean], dtype=np.float64)
    L = np.array([r["final_val_bpb"] for r in clean], dtype=np.float64)

    def loss(p):
        e, a, alpha, b, beta = p
        pred = np.exp(e) + np.exp(a) / N**alpha + np.exp(b) / D**beta
        r = np.log(pred) - np.log(L)
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
           "B": float(np.exp(b)), "beta": float(beta), "huber": float(best_v),
           "n_runs": len(clean), "implied_frontier_b": float(beta / (alpha + beta))}
    print(f"L(N,D) = {fit['E']:.4f} + {fit['A']:.3g}/N^{alpha:.3f} + {fit['B']:.3g}/D^{beta:.3f}"
          f"   (implied frontier b = β/(α+β) = {fit['implied_frontier_b']:.3f})")
    return fit


def plots(runs: list[dict], optima: list[dict], stab: list[dict], figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    by_budget: dict[float, list[dict]] = {}
    for r in runs:
        by_budget.setdefault(r["budget_flops"], []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (C, rs) in enumerate(sorted(by_budget.items())):
        rs = sorted(rs, key=lambda r: r["n_params_non_emb"])
        good = [r for r in rs if not r["diverged"]]
        bad = [r for r in rs if r["diverged"]]
        col = f"C{i}"
        ax.plot([r["n_params_non_emb"] for r in good], [r["final_val_bpb"] for r in good],
                "o-", color=col, label=f"C={C:.0e}")
        if bad:
            ax.plot([r["n_params_non_emb"] for r in bad], [r["final_val_bpb"] for r in bad],
                    "x", color=col, ms=9, mew=2)
    for r in stab:
        ax.plot(r["n_params_non_emb"], r["final_val_bpb"], "s", mfc="none",
                color="k", ms=7)
    if optima:
        interior = [o for o in optima if not o["censored"]]
        xs = [o["N_opt"] for o in interior]
        ys = [o["L_opt"] for o in interior]
        if any(o.get("L_std") for o in interior):
            ax.errorbar(xs, ys, yerr=[o.get("L_std") or np.nan for o in interior],
                        fmt="k--*", ms=11, capsize=3, label="fitted optima")
        else:
            ax.plot(xs, ys, "k--*", ms=11, label="fitted optima")
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("val bits/byte")
    ax.set_title("Galah IsoFLOP profiles (× diverged, □ stability annex)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "isoflop.png", dpi=160)

    interior = [o for o in optima if not o["censored"]]
    censored = [o for o in optima if o["censored"]]
    if len(interior) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        if any(o.get("N_opt_std") for o in interior):
            ax.errorbar([o["C"] for o in interior], [o["N_opt"] for o in interior],
                        yerr=[o.get("N_opt_std") or np.nan for o in interior],
                        fmt="o-", capsize=3, label="interior optima")
        else:
            ax.plot([o["C"] for o in interior], [o["N_opt"] for o in interior], "o-",
                    label="interior optima")
        if censored:
            ax.plot([o["C"] for o in censored], [o["N_opt"] for o in censored], "v",
                    mfc="none", label="censored (upper bound)")
        ax.set_xlabel("compute C (FLOPs)")
        ax.set_ylabel("N_opt")
        ax.set_title("compute-optimal frontier")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "frontier.png", dpi=160)
    print(f"figures → {figs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--stab", type=Path, default=Path("runs_stab"),
                    help="stability-annex runs; plotted, never fit")
    ap.add_argument("--figs", type=Path, default=Path("figures"))
    args = ap.parse_args()

    all_runs = load_runs(args.runs)
    main, repeats = partition_runs(all_runs)
    diverged = [r["name"] for r in main if r["diverged"]]
    print(f"{len(main)} sweep runs · {len(repeats)} seed-repeats/suffix · "
          f"{len(diverged)} diverged (excluded from fits): {diverged}")
    optima = isoflop_optima(main)
    stab = load_runs(args.stab) if args.stab.exists() else []
    stab_repeats = [r for r in stab if SEED_SUFFIX.search(r.get("name", ""))]
    attach_seed_errors(optima, main, repeats + stab_repeats)
    frontier = frontier_fit(optima)
    fit = parametric_fit(main) if len(main) >= 8 else None
    if stab:
        print(f"{len(stab)} stability-annex runs overlaid (not fit)")
    out = {"optima": optima, "frontier": frontier, "parametric": fit,
           "excluded_diverged": diverged,
           "stab_runs": [{k: r.get(k) for k in
                          ("name", "n_params_non_emb", "tokens", "final_val_bpb",
                           "lr_scale", "seed", "diverged")} for r in stab]}
    (args.runs / "fits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    plots(main, optima, stab + repeats, args.figs)


if __name__ == "__main__":
    main()
