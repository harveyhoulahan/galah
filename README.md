# galah

Small language models, scaled properly, shipped to a browser.

Galah is a family of byte-level GPTs (0.3M – 69M non-embedding parameters)
and a Chinchilla-style scaling study with one twist: the models have to
*ship* — into a browser tab, over a portfolio site's bandwidth budget, running
on whatever GPU the visitor brought. The paper asks where the compute-optimal
frontier moves when the constraint is deployment, not training FLOPs — and
answers by putting the winner live at [hjhportfolio.com](https://hjhportfolio.com).

Byte-level on purpose: no tokenizer to train or ship, embedding params stay
negligible at the smallest rungs (keeps 6·N·D accounting honest), and the
downstream terminal finetune inherits typo robustness for free.

## Layout

```
galah/model.py         the decoder + the size ladder (vanilla by design)
galah/data.py          FineWeb-Edu → flat uint8 memmaps
galah/train.py         one FLOP-budgeted run, fixed recipe
galah/sweep.py         IsoFLOP ladder over budgets, resumable
galah/seed_repeats.py  extra seeds at the interior per-budget optima
galah/fit.py           optima, frontier, parametric L(N,D) — the paper's curves
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m galah.data prepare --source fineweb-edu --gb 12
python -m galah.sweep --budgets 1e15,3e15,1e16,3e16,1e17,3e17 --compile
python -m galah.seed_repeats --compile
python -m galah.fit
```

The default sweep is sized for a single large consumer/workstation GPU
(trained on a 96 GB card; a 16 GB card runs everything below C=3e17). Every
run writes `runs/<name>/final.json`; re-launching the sweep skips finished
runs, so it survives interruption. `galah.seed_repeats` re-runs only the
interior per-budget optima (the configs the frontier slope actually depends
on) at extra seeds, and likewise skips any run whose `final.json` exists.

Recipe notes: AdamW(0.9, 0.95), wd 0.1, cosine to 10% with 2% warmup, bf16,
global batch 128 KiB of bytes, lr = 3e-3·(128/d)^0.5. The recipe is frozen
across all runs — (N, D) is the only thing the study varies. Per-run lr
tuning (as in Hoffmann et al.) is the main simplification; stated in the paper.

## Status

- [x] harness (model / data / train / sweep / fit)
- [x] sweep on lychee
- [x] paper: deployment-optimal correction + fits
- [ ] WebGPU (WGSL) inference runtime, int8
- [ ] terminal finetune (context-aware shell brain)
