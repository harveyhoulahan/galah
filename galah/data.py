"""Corpus preparation and batch loading.

`python -m galah.data prepare` streams FineWeb-Edu (sample-10BT) from
HuggingFace, concatenates documents with a 0x00 separator byte, and writes
flat uint8 memmaps: data/train.bin and data/val.bin. Byte-level means this IS
the tokenization — there is nothing else to version.

The loader serves random crops from the memmap; at these model sizes the GPU
is the bottleneck, not IO, so nothing fancier than pinned memory is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SEP = 0  # document separator byte


def prepare(out_dir: Path, gb: float, source: str, val_mb: int = 64) -> None:
    from datasets import load_dataset  # deferred: heavy import
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    if source == "fineweb-edu":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    elif source == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    else:
        raise SystemExit(f"unknown source '{source}'")

    target = int(gb * 1e9)
    buf = bytearray()
    with tqdm(total=target, unit="B", unit_scale=True, desc=f"{source} → bytes") as bar:
        for ex in ds:
            b = ex["text"].encode("utf-8", errors="ignore")
            buf += b
            buf.append(SEP)
            bar.update(len(b) + 1)
            if len(buf) >= target:
                break

    val_n = min(val_mb * 1_000_000, len(buf) // 20)
    train, val = bytes(buf[:-val_n]), bytes(buf[-val_n:])
    (out_dir / "train.bin").write_bytes(train)
    (out_dir / "val.bin").write_bytes(val)
    print(f"wrote {len(train)/1e9:.2f} GB train · {len(val)/1e6:.0f} MB val → {out_dir}")


class ByteBatches:
    """Random-crop batch server over a uint8 memmap."""

    def __init__(self, path: Path, seq_len: int, device: str, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.seq = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)
        if len(self.data) < seq_len + 1:
            raise SystemExit(f"{path} smaller than one sequence")

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.rng.integers(0, len(self.data) - self.seq - 1, size=batch)
        x = np.stack([self.data[i : i + self.seq] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + self.seq + 1] for i in ix]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.startswith("cuda"):
            return xt.pin_memory().to(self.device, non_blocking=True), yt.pin_memory().to(self.device, non_blocking=True)
        return xt.to(self.device), yt.to(self.device)

    @property
    def n_bytes(self) -> int:
        return len(self.data)


def main() -> None:
    try:
        from setproctitle import setproctitle
        setproctitle("train-worker-data")  # shared box: keep argv off other users' ps/btop
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="python -m galah.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="download + pack a byte corpus")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--gb", type=float, default=12.0, help="corpus size in GB of UTF-8 bytes")
    p.add_argument("--source", choices=["fineweb-edu", "tinystories"], default="fineweb-edu")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.out, args.gb, args.source)


if __name__ == "__main__":
    sys.exit(main())
