# analysis

Figures for the write-up. Everything here is derived; nothing here is a source of
truth. The sources of truth are the CSVs in `benchmarks/results/`.

## Regenerate the figures

```bash
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  .venv/bin/python analysis/plots.py
```

Reads four CSVs:

| CSV | What it holds |
|---|---|
| `scale_sweep_physics_only.csv` | end-to-end, no touch |
| `scale_sweep_splat.csv` | end-to-end, dense torch splat encoder |
| `scale_sweep_splat_triton.csv` | end-to-end, Triton kernel |
| `kernel_bench.csv` | encoder only: eager vs `torch.compile` vs Triton |

writes eight PNGs at 150 dpi into `analysis/plots/`, and prints the derived
headline numbers so a figure can be checked against the data without opening an
image:

```
crossover N (torch)       = 256
linear knee (physics)     = 128
encoder share at N=4,096  = 77.1%  (torch)
                          = 6.5%  (Triton)
Amdahl ceiling            = 4.37x
cost of touch at N=4,096 = 4.60x -> 1.08x
end-to-end gain           = 4.25x  (97% of the ceiling)
kernel at N=4,096         = 0.582 ms  (188x eager, 78x torch.compile)
torch.compile alone       = 2.41x  (-> 1.82x end to end)
Amdahl: 30x kernel        = 3.93x  vs 4.29x at 188x
tactile stage vs kernel   = 2.393 ms vs 0.582 ms  (76% pre-processing, inferred)
```

The memory cap is habit, not necessity — matplotlib on four small CSVs needs
almost nothing — but this box has unified CPU/GPU memory, so nothing here runs
uncapped (`PROJECT_LOG.md` §1.1). `matplotlib.use("Agg")` is set in the script;
there is no display on this machine.

## The figures

The first four are the pre-kernel story; the last four are the kernel result.

| File | Shows |
|---|---|
| `plots/A_bottleneck_migration.png` | physics ms/step and torch encoder ms/step vs N on log-log, with the crossover marked. |
| `plots/B_throughput_scaling.png` | env-steps/s vs N, no touch and torch splat, against an ideal-linear reference. |
| `plots/C_encoder_share.png` | torch encoder share of step time, 0–100%, with the 50% line and the Amdahl ceiling. |
| `plots/D_slowdown_factor.png` | physics-only throughput ÷ torch splat throughput. |
| **`plots/E_touch_is_free.png`** | **the payoff.** Throughput vs N for all three configurations. The Triton curve rejoins the no-touch curve; the torch curve does not. Cost of touch annotated at the top rung. |
| **`plots/F_encoder_three_ways.png`** | encoder-only ms/call vs N for eager, `torch.compile` and Triton, log-y. `torch.compile` is on the plot because it is the honest baseline — one line of code, no arithmetic changed — and the kernel is reported against it as well as against eager. |
| **`plots/G_migration_undone.png`** | figure A plus the Triton tactile stage, which never crosses the physics line. Also carries the *second* migration: the stage line sits well above the kernel-alone line, and the gap is pre-processing. |
| **`plots/H_amdahl.png`** | end-to-end speedup vs encoder speedup against the 4.37× ceiling, with `torch.compile`, a hypothetical 30× kernel and the measured 188× marked. The point is the flatness on the right: 30× and 188× land 0.36× apart. |

Two things on the figures are worth reading carefully rather than skimming:

- **G's "~76% pre-processing" is inferred, not instrumented.** It is
  `1 − (kernel ms ÷ stage ms)` from two separately measured numbers taken on
  different input paths (in-harness vs the encoder-only bench). The figure says
  "inferred" on its face for that reason. See README §4.8.
- **H's ceiling and curve come from the *pre-kernel* encoder share** (77.1%,
  registered before the kernel existed), while the star is the measured
  end-to-end ratio from the two fused-timing sweeps. That they agree — 4.29×
  predicted, 4.25× measured — is the check, so the two must keep coming from
  different places.

## Rules the script follows

- **No hardcoded results.** The crossover env count, the linear-scaling knee, the
  final throughputs, the cost of touch, the kernel speedups, the traffic figures,
  the Amdahl ceiling and the whole Amdahl curve are computed from the CSVs at plot
  time and interpolated into the titles and annotations. Re-run the sweeps and the
  figures — captions included — follow. A figure whose caption has drifted from
  its data is worse than no figure. The one number typed in by hand is the
  *hypothetical* 30× marker in H, which is a counterfactual and has no CSV.
- **Failed rungs are not plotted.** Rows with `ok != True` are the memory ceiling,
  a result in their own right, and are excluded from the curves rather than drawn
  as zeros. `kernel_bench.csv` has no `ok` column — it is not a sweep to
  failure — and the filter is skipped when the column is absent rather than
  assumed present.
- **Ratios are only taken between matching env counts.** `align()` intersects the
  env ladders before dividing. The three sweeps happen to share all 13 rungs, but
  that is a fact about the data, not a guarantee.
- **Titles state the finding**, not the variables; the grey lines beneath give the
  measurement conditions (scene, contacts per env, contact overflow, which timing
  pass).
- **Colour is never the only cue**, and it means the same thing everywhere:
  blue = physics / no touch, orange = the dense torch encoder, sky = the same
  encoder under `torch.compile`, green = the Triton kernel, purple = a derived
  ratio. Okabe-Ito hues (deuteranope/protanope-safe), and each series also differs
  in line style and marker.
- Which timing pass each figure uses is stated on the figure: A, C and G come from
  the stage-synchronised pass (the only one that separates physics from encoder,
  and an upper bound on the fused cost); B, D and E come from the fused pass; F is
  encoder-only with synthetic contacts and no physics at all.

## Adding a figure

Add a `fig_*(...)` function taking whichever loaded CSVs it needs, call it from
`main()`, and use `style_axes`, `titles` and `save` so it matches the others. If a
number belongs in the title, compute it from the arrays — do not type it in.
