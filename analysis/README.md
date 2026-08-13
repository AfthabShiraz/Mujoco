# analysis

Figures for the write-up. Everything here is derived; nothing here is a source of
truth. The sources of truth are the two CSVs in `benchmarks/results/`.

## Regenerate the figures

```bash
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  .venv/bin/python analysis/plots.py
```

Reads `benchmarks/results/scale_sweep_physics_only.csv` and
`scale_sweep_splat.csv`, writes four PNGs at 150 dpi into `analysis/plots/`, and
prints the derived headline numbers so a figure can be checked against the data
without opening an image:

```
crossover N            = 256
linear knee (physics)  = 128
encoder share at N=4,096 = 77.1%
Amdahl ceiling         = 4.37x
slowdown at N=4,096     = 4.60x
```

The memory cap is habit, not necessity — matplotlib on two small CSVs needs
almost nothing — but this box has unified CPU/GPU memory, so nothing here runs
uncapped (`PROJECT_LOG.md` §1.1). `matplotlib.use("Agg")` is set in the script;
there is no display on this machine.

## The figures

| File | Shows |
|---|---|
| `plots/A_bottleneck_migration.png` | physics ms/step and encoder ms/step vs N on log-log, with the crossover marked. The centrepiece. |
| `plots/B_throughput_scaling.png` | env-steps/s vs N for both configurations, against an ideal-linear reference. |
| `plots/C_encoder_share.png` | encoder share of step time, 0–100%, with the 50% line and the Amdahl ceiling. |
| `plots/D_slowdown_factor.png` | physics-only throughput ÷ splat throughput. |

## Rules the script follows

- **No hardcoded results.** The crossover env count, the linear-scaling knee, the
  final throughputs, the slowdown factor and the Amdahl ceiling are all computed
  from the CSVs at plot time and interpolated into the titles and annotations.
  Re-run the sweeps and the figures — captions included — follow. A figure whose
  caption has drifted from its data is worse than no figure.
- **Failed rungs are not plotted.** Rows with `ok != True` are the memory ceiling,
  a result in their own right, and are excluded from the curves rather than drawn
  as zeros.
- **Titles state the finding**, not the variables; the grey line beneath gives the
  measurement conditions (scene, contacts per env, contact overflow).
- **Colour is never the only cue.** Okabe-Ito hues (deuteranope/protanope-safe),
  and each series also differs in line style and marker.
- Which timing pass each figure uses is stated on the figure: A and C come from
  the stage-synchronised pass (the only one that separates physics from encoder,
  and an upper bound on the fused cost); B and D come from the fused pass.

## Adding a figure

Add a `fig_*(phys, splat)` function, call it from `main()`, and use `style_axes`,
`titles` and `save` so it matches the others. If a number belongs in the title,
compute it from the arrays — do not type it in.
