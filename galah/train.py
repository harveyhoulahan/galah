"""Single training run, FLOP-budgeted.

The sweep hands this script a model rung and a compute budget C; it derives
the step count from C / (flops_per_token · tokens_per_step) and trains with a
fixed recipe (AdamW, cosine to 10%, 2% warmup, bf16 autocast, grad clip 1.0).
The recipe is deliberately identical across every run in the study — the only
thing that varies between runs is (N, D).

Learning rate follows a width rule, lr = 3e-3 · (128 / d_model)^0.5, so rungs
don't need individual tuning. This is the largest methodological simplification
vs Hoffmann et al. (who tuned per-run); it is stated in the paper.

Outputs per run, under runs/<name>/:
  log.jsonl   one line per log interval (step, loss, val bits/byte, lr, flops)
  final.json  config + budget + final smoothed train loss + final val bits/byte
  model.pt    final weights (fp32 state_dict)

Usage (normally via galah.sweep):
  python -m galah.train --rung galah-0.8m --budget 1e16 --data data/
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import ByteBatches
from .model import LADDER, Galah

try:
    from setproctitle import setproctitle
except ImportError:
    def setproctitle(_: str) -> None:  # shared boxes without the package: no-op
        pass


def cosine_lr(step: int, total: int, base: float, warmup: int, floor_frac: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def val_bits_per_byte(model: Galah, batches: ByteBatches, iters: int = 20, batch: int = 32) -> float:
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = batches.sample(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters / math.log(2)  # nats → bits


def main() -> None:
    # Shared box: hide the project name and hyperparameters from other users'
    # ps/btop, which read argv by default. Doesn't hide cwd or file contents
    # (those already need ptrace/root to see); just the command line itself.
    setproctitle("train-worker")

    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(LADDER.keys()))
    ap.add_argument("--budget", type=float, required=True, help="training FLOPs, e.g. 1e16")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--tokens-per-step", type=int, default=131072, help="global batch in bytes")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiplier on the width-rule lr (stability-annex runs only; 1.0 = frozen recipe)")
    ap.add_argument("--suffix", default="", help="appended to the run name, e.g. -lr0.5")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=0, help="0 = auto (~10 evals per run)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = LADDER[args.rung]
    cfg.seq_len = args.seq
    model = Galah(cfg).to(device)
    if args.compile:
        model = torch.compile(model)

    fpt = cfg.flops_per_token()
    steps = max(50, int(args.budget / (fpt * args.tokens_per_step)))
    tokens_total = steps * args.tokens_per_step
    batch = max(1, args.tokens_per_step // args.seq)
    base_lr = 3e-3 * (128 / cfg.d_model) ** 0.5 * args.lr_scale
    warmup = max(20, steps // 50)
    val_every = args.val_every or max(50, steps // 10)

    name = f"{args.rung}_C{args.budget:.0e}".replace("+", "") + args.suffix
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "log.jsonl", "w", encoding="utf-8")

    n = cfg.n_params_non_emb
    print(f"{name}: N={n/1e6:.2f}M · D={tokens_total/1e9:.2f}GB · D/N={tokens_total/n:.0f} "
          f"· {steps} steps · batch {batch}x{args.seq} · lr {base_lr:.1e} · {device}")

    train_b = ByteBatches(args.data / "train.bin", args.seq, device, seed=args.seed)
    val_b = ByteBatches(args.data / "val.bin", args.seq, device, seed=args.seed + 1)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1)
    ema_loss, t0 = None, time.time()
    val_bpb = float("nan")

    for step in range(steps):
        lr = cosine_lr(step, steps, base_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_b.sample(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        l = loss.item()
        ema_loss = l if ema_loss is None else 0.98 * ema_loss + 0.02 * l
        if step % val_every == 0 or step == steps - 1:
            val_bpb = val_bits_per_byte(model, val_b)
        if step % args.log_every == 0 or step == steps - 1:
            rec = {
                "step": step, "loss": round(l, 4), "ema": round(ema_loss, 4),
                "val_bpb": round(val_bpb, 4), "lr": round(lr, 6),
                "flops": step * fpt * args.tokens_per_step, "sec": round(time.time() - t0, 1),
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            tput = (step + 1) * args.tokens_per_step / (time.time() - t0)
            print(f"  {step:>6}/{steps}  loss {ema_loss:.4f}  val {val_bpb:.4f} bpb  "
                  f"{tput_fmt(tput)}  mfu~{mfu(fpt, tput, device):.0%}", flush=True)

    final = {
        "name": name, "rung": args.rung, "config": cfg.to_dict(),
        "n_params_non_emb": n, "n_params_total": cfg.n_params_total,
        "budget_flops": args.budget, "flops_per_token": fpt,
        "tokens": tokens_total, "steps": steps,
        "seed": args.seed, "lr_scale": args.lr_scale, "base_lr": base_lr,
        "final_train_loss_ema": round(ema_loss, 5),
        "final_val_bpb": round(val_bits_per_byte(model, val_b, iters=60), 5),
        "wall_sec": round(time.time() - t0, 1),
    }
    (run_dir / "final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw.state_dict(), run_dir / "model.pt")
    log_f.close()
    print(f"done: val {final['final_val_bpb']} bpb · {final['wall_sec']:.0f}s → {run_dir}")


def tput_fmt(tput: float) -> str:
    return f"{tput/1e6:.2f} MB/s" if tput > 1e6 else f"{tput/1e3:.0f} KB/s"


def mfu(flops_per_token: float, tput: float, device: str) -> float:
    peak = 200e12 if device == "cuda" else 1e12  # rough bf16 dense; report only
    return flops_per_token * tput / peak


if __name__ == "__main__":
    main()
