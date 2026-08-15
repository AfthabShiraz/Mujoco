"""Regenerate every figure in the write-up from the banked benchmark CSVs.

Nothing here is hardcoded from a previous run: every number that appears on a
figure -- the crossover env count, the linear-scaling knee, the slowdown factor,
the Amdahl ceiling, the kernel speedups -- is derived from

    benchmarks/results/scale_sweep_physics_only_dt01.csv  (no touch)
    benchmarks/results/scale_sweep_splat_dt01.csv         (torch splat encoder)
    benchmarks/results/scale_sweep_splat_triton_dt01.csv  (Triton splat encoder)
    benchmarks/results/kernel_bench.csv                   (encoder only: eager /
                                                           torch.compile / Triton)

The `_dt01` suffix comes from `SWEEP_SUFFIX` below and is the default: those
sweeps ran at sim_dt=0.01, matching `leapXelaMjLab/env_cfg.py`. Set
`SWEEP_SUFFIX=""` to plot the superseded dt=0.001 sweeps instead.

at plot time. Re-run the sweeps and the figures (and their titles) follow. That
is deliberate: a figure whose caption drifts from its data is worse than no
figure.

    .venv/bin/python analysis/plots.py            # -> analysis/plots/*.png

The box has unified CPU/GPU memory (PROJECT_LOG §1.1), so even this runs under
a cap by habit:

    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
      .venv/bin/python analysis/plots.py

Colours are Okabe-Ito, which is deuteranope/protanope-safe, and every series
also differs in line style and marker so identity never rests on hue alone.
"""

from __future__ import annotations

import csv
import os
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless box: no display, no interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
# report.tex looks in figures/ (the template's convention). Keep real copies
# there rather than a symlink -- Overleaf and some TeX setups will not follow a
# symlinked directory, and the report then silently renders without figures.
FIGURES_DIR = ROOT / "figures"
OUT = ROOT / "analysis" / "plots"
DPI = 150

# Which sweep set to plot. The `_dt01` CSVs ran at sim_dt=0.01, matching the
# supervisor's env_cfg.py; the unsuffixed ones inherited timestep=0.001 from the
# scene's included XML and are kept only for reference (see HYPOTHESES.md D9).
# Override with SWEEP_SUFFIX="" to regenerate the older figures.
SWEEP_SUFFIX = os.environ.get("SWEEP_SUFFIX", "_dt01")

PHYSICS_CSV = RESULTS / f"scale_sweep_physics_only{SWEEP_SUFFIX}.csv"
SPLAT_CSV = RESULTS / f"scale_sweep_splat{SWEEP_SUFFIX}.csv"
TRITON_CSV = RESULTS / f"scale_sweep_splat_triton{SWEEP_SUFFIX}.csv"
KERNEL_CSV = RESULTS / "kernel_bench.csv"

# The other two rigid representations, for the cost-of-sensing comparison (I).
# Flex is deliberately absent: it is a different body tree in a different model
# lineage, so its env-steps/s is NOT a like-for-like bar and HYPOTHESES.md H4
# says in terms that it must not be plotted as one. It stays in the report's
# table, with its footnote, where the caveat can travel with the number.
TOUCH_CSV = RESULTS / f"scale_sweep_touch{SWEEP_SUFFIX}.csv"
GRID_CSV = RESULTS / f"scale_sweep_touchgrid_object{SWEEP_SUFFIX}.csv"

# The contact-budget scans (J). Two series, same conditions except for the
# geometry, because the claim is that the cost is the allocation and nothing
# else -- if that is right, the two slopes must agree.
BUDGET_NONE_CSV = RESULTS / "contact_budget_none_n1024.csv"
BUDGET_GRID_CSV = RESULTS / "contact_budget_grid_off_n1024.csv"

# Okabe-Ito. BLUE/ORANGE are the two that must never be confused (they carry the
# crossover), and they are the best-separated pair in the set under CVD.
# Semantics are fixed across every figure: blue = physics / no touch,
# orange = the torch splat encoder, green = the Triton kernel.
BLUE = "#0072B2"    # physics / no touch
ORANGE = "#E69F00"  # torch splat encoder (eager)
GREEN = "#009E73"   # Triton kernel
PURPLE = "#CC79A7"  # derived ratios (slowdown)
SKY = "#56B4E9"     # torch.compile -- the honest middle baseline
GREY = "#666666"    # reference lines, annotation
FAINT = "#BBBBBB"

XTICKS = [1, 4, 16, 64, 256, 1024, 4096]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read one sweep CSV, keep only the rows that succeeded, sort by env count.

    Columns are returned as float arrays; anything non-numeric (`note`, `device`,
    the empty cells of the physics-only file) becomes NaN and is simply not used.
    A failed rung -- the memory ceiling -- is a legitimate row in the sweep files
    and must not be plotted as a data point, hence the `ok` filter. `kernel_bench`
    has no `ok` column (it is not a sweep to failure), so the filter is skipped
    when the column is absent rather than assumed present.
    """
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "ok" in rows[0]:
        rows = [r for r in rows if r["ok"].strip().lower() == "true"]
    if not rows:
        raise SystemExit(f"{path} has no successful rows")
    rows.sort(key=lambda r: int(r["num_envs"]))

    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[key] = np.asarray(vals, dtype=float)
    return out


def load_budget(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read one contact-budget scan, sorted by the budget rather than by N.

    `load` sorts on `num_envs`, which is constant across a budget scan -- the
    whole point is that N is held fixed while `nconmax_per_env` varies. Sorting
    on the wrong key would leave the fit correct (least squares does not care
    about order) but draw the connecting line as a zigzag, so it is worth its own
    loader rather than a flag.
    """
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r.get("ok", "true").strip().lower() == "true"]
    if not rows:
        raise SystemExit(f"{path} has no successful rows")
    rows.sort(key=lambda r: int(r["nconmax_per_env"]))

    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[key] = np.asarray(vals, dtype=float)
    return out


def fit_affine(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Least-squares `y = a + b*x`, returning (a, b, R^2).

    Mirrors `benchmarks/contact_budget_sweep.py:fit_affine` so the figure's
    printed law and the sweep's own are the same computation, not two that happen
    to agree.
    """
    b, a = np.polyfit(xs, ys, 1)
    pred = a + b * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return float(a), float(b), 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def at_envs(d: dict[str, np.ndarray], n: float, col: str) -> float:
    """One column's value at a given env count, or NaN if that rung is absent."""
    hit = np.flatnonzero(d["num_envs"] == n)
    return float(d[col][hit[0]]) if hit.size else float("nan")


def matched(a: dict[str, np.ndarray], b: dict[str, np.ndarray]):
    """Indices into `a` and `b` for the env counts both sweeps completed."""
    common = np.intersect1d(a["num_envs"], b["num_envs"])
    ia = np.searchsorted(a["num_envs"], common)
    ib = np.searchsorted(b["num_envs"], common)
    return common, ia, ib


def align(*ds: dict[str, np.ndarray]):
    """Common env counts across any number of sweeps, plus an index per sweep.

    The three sweeps happen to share all 13 rungs, but that is a fact about the
    data rather than a guarantee, and a ratio between mismatched rungs would be
    silently wrong rather than obviously wrong.
    """
    common = ds[0]["num_envs"]
    for d in ds[1:]:
        common = np.intersect1d(common, d["num_envs"])
    idx = [np.searchsorted(d["num_envs"], common) for d in ds]
    return common, idx


def amdahl(frac: float, speedup):
    """End-to-end speedup from making a stage of share `frac` go `speedup` times
    faster. `speedup = inf` gives the ceiling 1/(1 - frac)."""
    speedup = np.asarray(speedup, dtype=float)
    return 1.0 / ((1.0 - frac) + frac / speedup)


def crossover_n(splat: dict[str, np.ndarray]) -> float:
    """Smallest env count at which the encoder costs more than the physics step.

    Read off the split-timing pass, the only measurement that separates the two.
    """
    over = splat["num_envs"][splat["encoder_ms_per_step"] > splat["physics_ms_per_step"]]
    if over.size == 0:
        raise SystemExit("no crossover in the data")
    return float(over[0])


def linear_knee(d: dict[str, np.ndarray], tol: float = 1.15) -> float:
    """Largest N whose ms/step is still within `tol` of the single-env ms/step.

    While ms/step is flat, N environments cost what one costs -- i.e. throughput
    is linear in N. The first rung that breaks that is where the GPU saturates.
    """
    base = d["ms_per_step"][0]
    ok = d["num_envs"][d["ms_per_step"] <= tol * base]
    return float(ok[-1])


# --------------------------------------------------------------------------- #
# Shared styling
# --------------------------------------------------------------------------- #


def style_axes(ax, xlabel="environments (N)", logx=True, logy=False,
               xticks=None, xticklabels=None, logbase=2):
    if logx:
        ax.set_xscale("log", base=logbase)
        ticks = XTICKS if xticks is None else xticks
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks]
                           if xticklabels is None else xticklabels)
        ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", color=FAINT, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=GREY, labelcolor="black")


def titles(ax, finding: str, context: str):
    """Headline states the finding; the small lines under it state the setup.

    The context block is drawn upward from the axes top, so the title's padding
    has to clear it -- hence the line count in the pad.
    """
    n_lines = context.count("\n") + 1
    ax.set_title(finding, fontsize=12, fontweight="bold", loc="left",
                 pad=14 + 11 * n_lines)
    ax.text(0.0, 1.015, context, transform=ax.transAxes, fontsize=8.5,
            color=GREY, ha="left", va="bottom", linespacing=1.35)


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def setup_note(splat: dict[str, np.ndarray]) -> str:
    contacts = float(np.nanmean(splat["contacts_per_env"]))
    overflow = int(np.nansum(splat["overflow_worlds"]))
    return (f"MuJoCo Warp, DGX Spark (GB10, aarch64) · cube-reorient scene, hand at 60% closure · "
            # `overflow_worlds` counts worlds raising ANY Warp overflow bit, not
            # contacts specifically. At sim_dt=0.01 the flag that fires is NEFC
            # (constraint rows, njmax=120 -> wants 124); contacts sit at ~7/env
            # against a 48 cap. Labelling this "contact overflow" would be wrong.
            f"~{contacts:.1f} contacts/env · worlds w/ any overflow flag: {overflow}")


# --------------------------------------------------------------------------- #
# A. The migration figure
# --------------------------------------------------------------------------- #


def fig_a(phys, splat):
    n = splat["num_envs"]
    p_ms = splat["physics_ms_per_step"]
    e_ms = splat["encoder_ms_per_step"]
    xover = crossover_n(splat)
    i_x = int(np.searchsorted(n, xover))
    i_max = int(np.argmax(n))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    style_axes(ax, logy=True)

    # Encoder-bound regime, shaded from the crossover to the end of the sweep.
    ax.axvspan(xover, n[i_max] * 1.35, color=ORANGE, alpha=0.06, lw=0)

    ax.plot(n, p_ms, color=BLUE, lw=2.0, marker="o", ms=5.5,
            label="physics step (MuJoCo Warp)")
    ax.plot(n, e_ms, color=ORANGE, lw=2.0, ls="--", marker="s", ms=5.5,
            label="splat encoder (368 taxels)")

    ax.axvline(xover, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.annotate(
        f"crossover, N = {xover:,.0f}\n"
        f"encoder {e_ms[i_x]:.1f} ms > physics {p_ms[i_x]:.1f} ms",
        xy=(xover, e_ms[i_x]), xytext=(xover * 1.6, e_ms[i_x] * 0.30),
        fontsize=8.5, color="black", ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=4, shrinkB=4),
    )

    ax.annotate(f"{e_ms[i_max]:.0f} ms", xy=(n[i_max], e_ms[i_max]),
                xytext=(6, 2), textcoords="offset points",
                fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.annotate(f"{p_ms[i_max]:.0f} ms", xy=(n[i_max], p_ms[i_max]),
                xytext=(6, -2), textcoords="offset points",
                fontsize=8.5, color=BLUE, fontweight="bold")

    ax.set_ylabel("time per simulation step (ms)")
    ax.set_xlim(0.75, n[i_max] * 2.2)
    titles(
        ax,
        f"The bottleneck moves: touch overtakes physics at N = {xover:,.0f}",
        "per-stage split, measured with a synchronisation between the physics step and the encoder\n"
        + setup_note(splat),
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "A_bottleneck_migration.png")


# --------------------------------------------------------------------------- #
# B. Throughput
# --------------------------------------------------------------------------- #


def fig_b(phys, splat):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    style_axes(ax, logy=True)

    # Ideal linear scaling, anchored on the single-env physics measurement: what
    # throughput would be if N envs cost exactly what one env costs.
    ideal_n = np.array([phys["num_envs"][0], phys["num_envs"][-1] * 1.2])
    ideal = phys["env_steps_per_sec"][0] * ideal_n / phys["num_envs"][0]
    ax.plot(ideal_n, ideal, color=FAINT, lw=1.2, ls=(0, (2, 2)), zorder=1,
            label="ideal linear scaling")

    ax.plot(phys["num_envs"], phys["env_steps_per_sec"], color=BLUE, lw=2.0,
            marker="o", ms=5.5, label="physics only")
    ax.plot(splat["num_envs"], splat["env_steps_per_sec"], color=ORANGE, lw=2.0,
            ls="--", marker="s", ms=5.5, label="physics + splat encoder")

    knee = linear_knee(phys)
    ax.axvline(knee, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.annotate(f"GPU saturates\nN ≈ {knee:,.0f}", xy=(knee, phys["env_steps_per_sec"][0] * 1.5),
                xytext=(3, 0), textcoords="offset points",
                fontsize=8.5, color=GREY, ha="left", va="bottom")

    for d, colour in ((phys, BLUE), (splat, ORANGE)):
        i = int(np.argmax(d["num_envs"]))
        ax.annotate(f"{d['env_steps_per_sec'][i]:,.0f}",
                    xy=(d["num_envs"][i], d["env_steps_per_sec"][i]),
                    xytext=(6, -3), textcoords="offset points",
                    fontsize=8.5, color=colour, fontweight="bold")

    ax.set_ylabel("throughput (env-steps / s)")
    ax.set_xlim(0.75, phys["num_envs"][-1] * 3.0)
    titles(
        ax,
        f"Physics scales linearly to N ≈ {knee:,.0f}; the tactile path stops scaling far earlier",
        setup_note(splat),
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "B_throughput_scaling.png")


# --------------------------------------------------------------------------- #
# C. Encoder share
# --------------------------------------------------------------------------- #


def fig_c(phys, splat):
    n = splat["num_envs"]
    share = 100.0 * splat["encoder_frac"]
    xover = crossover_n(splat)
    i_max = int(np.argmax(n))
    top = share[i_max]
    amdahl = 1.0 / (1.0 - top / 100.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    style_axes(ax)

    ax.axhline(50.0, color=GREY, lw=1.0, ls=(0, (4, 3)))
    ax.text(1.0, 51.5, "half of step time", fontsize=8.5, color=GREY, va="bottom")

    ax.plot(n, share, color=ORANGE, lw=2.2, marker="s", ms=5.5)
    ax.axvline(xover, color=GREY, lw=1.0, ls=":", zorder=1)

    i_x = int(np.searchsorted(n, xover))
    ax.plot([xover], [share[i_x]], marker="o", ms=11, mfc="none",
            mec=ORANGE, mew=1.8, zorder=4)
    ax.annotate(f"N = {xover:,.0f}: {share[i_x]:.1f}%",
                xy=(xover, share[i_x]), xytext=(-8, -26), textcoords="offset points",
                fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=0, shrinkB=8))
    ax.annotate(f"{top:.1f}% at N = {n[i_max]:,.0f}\nAmdahl ceiling {amdahl:.2f}×",
                xy=(n[i_max], top), xytext=(-6, 42), textcoords="offset points",
                fontsize=9, ha="right", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=2, shrinkB=6))

    ax.set_ylim(0, 100)
    ax.set_xlim(0.75, n[i_max] * 1.6)
    ax.set_ylabel("encoder share of step time (%)")
    titles(
        ax,
        f"Tactile encoding grows from ~{share[0]:.0f}% to {top:.0f}% of step time",
        f"share = encoder / (physics + encoder), split timing · eliminating the encoder entirely "
        f"would gain at most {amdahl:.2f}× end to end at N = {n[i_max]:,.0f}\n" + setup_note(splat),
    )
    save(fig, "C_encoder_share.png")


# --------------------------------------------------------------------------- #
# D. Slowdown
# --------------------------------------------------------------------------- #


def fig_d(phys, splat):
    common, ip, isp = matched(phys, splat)
    slowdown = phys["env_steps_per_sec"][ip] / splat["env_steps_per_sec"][isp]
    i_max = int(np.argmax(common))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    style_axes(ax)

    ax.axhline(1.0, color=GREY, lw=1.0, ls=(0, (4, 3)))
    ax.text(common[i_max] * 1.5, 1.06, "no cost (touch is free)", fontsize=8.5,
            color=GREY, va="bottom", ha="right")

    ax.plot(common, slowdown, color=PURPLE, lw=2.2, marker="D", ms=5.0)
    ax.annotate(f"{slowdown[i_max]:.2f}×", xy=(common[i_max], slowdown[i_max]),
                xytext=(-4, 8), textcoords="offset points",
                fontsize=10, color=PURPLE, fontweight="bold", ha="right")

    ax.set_ylim(0, max(5.0, slowdown.max() * 1.18))
    ax.set_xlim(0.75, common[i_max] * 1.6)
    ax.set_ylabel("throughput penalty (×)")
    titles(
        ax,
        f"Turning touch on costs {slowdown[0]:.2f}× at N = {common[0]:,.0f} "
        f"and {slowdown[i_max]:.2f}× at N = {common[i_max]:,.0f}",
        "physics-only throughput ÷ physics+splat throughput, fused timing (no inner synchronisation)\n"
        + setup_note(splat),
    )
    save(fig, "D_slowdown_factor.png")


# --------------------------------------------------------------------------- #
# E. The payoff: three configurations on one throughput plot
# --------------------------------------------------------------------------- #


def fig_e(phys, splat, tri):
    """Throughput vs N for no touch / torch splat / Triton splat.

    The point of the figure is a shape, not a number: the Triton curve rejoins
    the no-touch curve, and the torch curve does not.
    """
    common, (ip, isp, it) = align(phys, splat, tri)
    i_max = int(np.argmax(common))
    cost_torch = phys["env_steps_per_sec"][ip] / splat["env_steps_per_sec"][isp]
    cost_tri = phys["env_steps_per_sec"][ip] / tri["env_steps_per_sec"][it]
    gain = tri["env_steps_per_sec"][it] / splat["env_steps_per_sec"][isp]
    n_top = common[i_max]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    style_axes(ax, logy=True)

    ax.plot(phys["num_envs"], phys["env_steps_per_sec"], color=BLUE, lw=2.0,
            marker="o", ms=5.5, label="no touch (physics only)")
    ax.plot(splat["num_envs"], splat["env_steps_per_sec"], color=ORANGE, lw=2.0,
            ls="--", marker="s", ms=5.5, label="torch splat encoder")
    ax.plot(tri["num_envs"], tri["env_steps_per_sec"], color=GREEN, lw=2.0,
            ls=(0, (5, 1, 1, 1)), marker="^", ms=6.0, label="Triton splat kernel")

    # The gap that is left. Drawn rather than asserted.
    ax.annotate(
        "", xy=(n_top, phys["env_steps_per_sec"][ip][i_max]),
        xytext=(n_top, splat["env_steps_per_sec"][isp][i_max]),
        arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9, shrinkA=2, shrinkB=2),
    )
    ax.annotate(f"cost of touch\n{cost_torch[i_max]:.2f}× → {cost_tri[i_max]:.2f}×",
                xy=(n_top, (phys["env_steps_per_sec"][ip][i_max]
                            * splat["env_steps_per_sec"][isp][i_max]) ** 0.5),
                xytext=(-12, 0), textcoords="offset points",
                fontsize=8.5, color=GREY, ha="right", va="center",
                bbox=dict(fc="white", ec="none", pad=1.5))

    # The no-touch and Triton endpoints nearly coincide -- which is the finding --
    # so their labels have to be pushed apart by hand.
    for d, i, colour, dy in ((phys, ip, BLUE, 8), (tri, it, GREEN, -10),
                             (splat, isp, ORANGE, -3)):
        ax.annotate(f"{d['env_steps_per_sec'][i][i_max]:,.0f}",
                    xy=(n_top, d["env_steps_per_sec"][i][i_max]),
                    xytext=(7, dy), textcoords="offset points",
                    fontsize=8.5, color=colour, fontweight="bold")

    ax.set_ylabel("throughput (env-steps / s)")
    ax.set_xlim(0.75, n_top * 3.2)
    titles(
        ax,
        f"Touch is now nearly free: {cost_tri[i_max]:.2f}× at N = {n_top:,.0f}, "
        f"down from {cost_torch[i_max]:.2f}×",
        f"fused timing (no inner synchronisation) · the Triton curve rejoins the no-touch curve · "
        f"{gain[i_max]:.2f}× over the torch encoder at N = {n_top:,.0f}\n"
        + setup_note(tri),
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "E_touch_is_free.png")


# --------------------------------------------------------------------------- #
# F. Encoder cost, three ways
# --------------------------------------------------------------------------- #


def fig_f(kern):
    """Encoder-only ms/call for eager, torch.compile and Triton.

    `torch.compile` is on the plot because it is the honest baseline: it is one
    line of code, it changes no arithmetic, and any kernel that cannot beat it
    was not worth writing.
    """
    n = kern["num_envs"]
    i_max = int(np.argmax(n))
    n_top = n[i_max]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    # This sweep's rungs are a subset of the end-to-end ladder, so label the rungs
    # it actually has rather than the shared tick set.
    style_axes(ax, logy=True, xticks=[int(v) for v in n])

    ax.plot(n, kern["eager_ms"], color=ORANGE, lw=2.0, ls="--", marker="s", ms=5.5,
            label="dense torch encoder (eager)")
    ax.plot(n, kern["compile_ms"], color=SKY, lw=2.0, ls=(0, (1, 1.4)), marker="v",
            ms=5.5, label="same encoder, torch.compile")
    ax.plot(n, kern["triton_ms"], color=GREEN, lw=2.0, ls=(0, (5, 1, 1, 1)),
            marker="^", ms=6.0, label="Triton two-pass gather kernel")

    for key, colour in (("eager_ms", ORANGE), ("compile_ms", SKY),
                        ("triton_ms", GREEN)):
        ax.annotate(f"{kern[key][i_max]:,.3g} ms", xy=(n_top, kern[key][i_max]),
                    xytext=(7, -3), textcoords="offset points",
                    fontsize=8.5, color=colour, fontweight="bold")

    ax.annotate(
        f"{kern['speedup_vs_eager'][i_max]:,.0f}× vs eager\n"
        f"{kern['speedup_vs_compile'][i_max]:,.0f}× vs torch.compile",
        xy=(n_top, kern["triton_ms"][i_max] * 6.0),
        xytext=(-12, 0), textcoords="offset points",
        fontsize=9, color="black", ha="right", va="center", fontweight="bold",
    )

    ax.set_ylabel("encoder time per call (ms)")
    ax.set_xlim(n[0] * 0.7, n_top * 3.6)
    titles(
        ax,
        f"Encoder cost at N = {n_top:,.0f}: {kern['eager_ms'][i_max]:.1f} ms → "
        f"{kern['compile_ms'][i_max]:.1f} ms → {kern['triton_ms'][i_max]:.3f} ms",
        f"encoder only, synthetic contacts at the {kern['contacts_per_env_cap'][i_max]:.0f}-slot budget, "
        f"{kern['n_taxels'][i_max]:.0f} taxels · torch.compile is a one-line change and gives "
        f"{kern['compile_speedup_vs_eager'][i_max]:.2f}×\n"
        f"modelled traffic {kern['eager_bytes_mb'][i_max]:,.0f} MB → "
        f"{kern['triton_bytes_mb'][i_max]:,.0f} MB against a "
        f"{kern['floor_bytes_mb'][i_max]:,.0f} MB compulsory floor · runtime is data-dependent "
        f"(see README)",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "F_encoder_three_ways.png")


# --------------------------------------------------------------------------- #
# G. The migration, undone
# --------------------------------------------------------------------------- #


def fig_g(splat, tri, kern):
    """Figure A with the Triton stage added: the encoder never overtakes physics.

    Also carries the second migration. The Triton *stage* line sits well above
    the Triton *kernel* line, and the difference is pre-processing.
    """
    n = splat["num_envs"]
    p_ms = splat["physics_ms_per_step"]
    e_ms = splat["encoder_ms_per_step"]
    xover = crossover_n(splat)
    i_max = int(np.argmax(n))

    tn = tri["num_envs"]
    t_ms = tri["encoder_ms_per_step"]
    j_max = int(np.argmax(tn))
    peak_share = 100.0 * np.nanmax(tri["encoder_frac"])
    crossed = np.any(t_ms > tri["physics_ms_per_step"])

    # Second migration: the kernel is only part of the stage. Both numbers are
    # measured; the split between them is inferred from the pair, not instrumented.
    kn, (ik, it) = align(kern, tri)
    k_ms = kern["triton_ms"][ik]
    stage_ms = tri["encoder_ms_per_step"][it]
    pre_frac = 100.0 * (1.0 - k_ms[-1] / stage_ms[-1])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    style_axes(ax, logy=True)

    ax.axvspan(xover, n[i_max] * 1.35, color=ORANGE, alpha=0.06, lw=0)

    ax.plot(n, p_ms, color=BLUE, lw=2.0, marker="o", ms=5.5,
            label="physics step (MuJoCo Warp)")
    ax.plot(n, e_ms, color=ORANGE, lw=2.0, ls="--", marker="s", ms=5.5,
            label="torch splat encoder")
    ax.plot(tn, t_ms, color=GREEN, lw=2.0, ls=(0, (5, 1, 1, 1)), marker="^", ms=6.0,
            label="Triton tactile stage (kernel + pre-processing)")
    ax.plot(kn, k_ms, color=GREEN, lw=1.4, ls=":", marker="D", ms=4.0, alpha=0.85,
            label="Triton kernel alone")

    ax.axvline(xover, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.annotate(f"old crossover\nN = {xover:,.0f}", xy=(xover, e_ms[i_max]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8.5, color=GREY, ha="left", va="center")

    ax.annotate(f"{e_ms[i_max]:.0f} ms", xy=(n[i_max], e_ms[i_max]),
                xytext=(6, 2), textcoords="offset points",
                fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.annotate(f"{p_ms[i_max]:.0f} ms", xy=(n[i_max], p_ms[i_max]),
                xytext=(6, -2), textcoords="offset points",
                fontsize=8.5, color=BLUE, fontweight="bold")

    # The gap between the two green lines is the second migration.
    ax.annotate(
        "", xy=(kn[-1], stage_ms[-1]), xytext=(kn[-1], k_ms[-1]),
        arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.9, shrinkA=2, shrinkB=2),
    )
    # Placed in the empty band under the physics line rather than beside the
    # arrow: a label there would sit on top of both green curves.
    ax.text(n[0] * 1.06, (e_ms[0] * p_ms[0]) ** 0.5,
            f"at N = {kn[-1]:,.0f}: {stage_ms[-1]:.2f} ms stage,\n"
            f"{k_ms[-1]:.2f} ms kernel → ~{pre_frac:.0f}% of the\n"
            f"tactile stage is pre-processing\n(inferred, not instrumented)",
            fontsize=8.5, color="black", ha="left", va="center", linespacing=1.35)

    ax.set_ylabel("time per simulation step (ms)")
    ax.set_xlim(0.75, n[i_max] * 2.4)
    titles(
        ax,
        ("The migration is undone: the Triton stage never overtakes physics"
         if not crossed else
         "The Triton stage still overtakes physics at some N"),
        f"per-stage split, measured with a synchronisation between the physics step and the encoder · "
        f"the Triton stage peaks at {peak_share:.1f}% of step time\n"
        + setup_note(tri),
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "G_migration_undone.png")


# --------------------------------------------------------------------------- #
# H. Amdahl
# --------------------------------------------------------------------------- #


def fig_h(phys, splat, tri, kern):
    """End-to-end speedup vs encoder speedup, against the ceiling.

    The ceiling and the curve come from the *pre-kernel* encoder share, which was
    registered before the kernel existed. The measured point comes from the two
    fused-timing sweeps. They agree, which is the check worth having.
    """
    frac = float(splat["encoder_frac"][int(np.argmax(splat["num_envs"]))])
    n_top = float(np.max(splat["num_envs"]))
    ceiling = 1.0 / (1.0 - frac)

    k_i = int(np.argmax(kern["num_envs"]))
    s_triton = float(kern["speedup_vs_eager"][k_i])
    s_compile = float(kern["compile_speedup_vs_eager"][k_i])

    common, (isp, it) = align(splat, tri)
    measured = float(tri["env_steps_per_sec"][it][-1]
                     / splat["env_steps_per_sec"][isp][-1])

    xs = np.logspace(0, np.log10(s_triton * 3.0), 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ticks = [1, 2, 5, 10, 30, 100, int(round(s_triton))]
    style_axes(ax, xlabel="encoder speedup over the eager torch encoder (×)",
               logx=True, logy=False, logbase=10,
               xticks=ticks, xticklabels=[f"{t:g}×" for t in ticks])

    ax.plot(xs, amdahl(frac, xs), color=GREY, lw=1.8,
            label=f"Amdahl, encoder = {100 * frac:.1f}% of step time")
    ax.axhline(ceiling, color="black", lw=1.2, ls=(0, (4, 3)))
    ax.text(1.05, ceiling - 0.13, f"ceiling {ceiling:.2f}× — an infinitely fast encoder",
            fontsize=9, color="black", va="top", fontweight="bold")

    # The three points that make the diminishing-returns argument.
    marks = [
        (s_compile, f"torch.compile alone:\n{s_compile:.2f}× kernel → {amdahl(frac, s_compile):.2f}× end to end",
         SKY, 10, -22),
        (30.0, f"hypothetical 30×\n→ {amdahl(frac, 30.0):.2f}×", GREY, -6, -34),
        (s_triton, f"this kernel, {s_triton:.0f}×\n→ {amdahl(frac, s_triton):.2f}× predicted", GREEN, -8, -40),
    ]
    for x, label, colour, dx, dy in marks:
        y = float(amdahl(frac, x))
        ax.plot([x], [y], marker="o", ms=8, mfc="white", mec=colour, mew=2.0, zorder=5)
        ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color="black", ha="right" if dx < 0 else "left",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.8,
                                    shrinkA=0, shrinkB=6))

    ax.plot([s_triton], [measured], marker="*", ms=16, color=GREEN, zorder=6)
    ax.annotate(f"measured end to end: {measured:.2f}×\n"
                f"{100 * measured / ceiling:.0f}% of what was available",
                xy=(s_triton, measured), xytext=(-10, 26), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8, shrinkA=0, shrinkB=8))

    gap = float(amdahl(frac, s_triton) - amdahl(frac, 30.0))
    ax.set_ylim(0, ceiling * 1.28)
    ax.set_xlim(0.95, s_triton * 3.0)
    ax.set_ylabel("end-to-end speedup (×)")
    titles(
        ax,
        f"Amdahl is the constraint, not the kernel: {measured:.2f}× against a "
        f"{ceiling:.2f}× ceiling",
        f"ceiling and curve from the encoder's {100 * frac:.1f}% share at N = {n_top:,.0f}, registered "
        f"before the kernel existed\n"
        f"a {s_triton:.0f}× kernel and a hypothetical 30× kernel land {gap:.2f}× apart end to end",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "H_amdahl.png")


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# I. The cost of sensing: every comparable representation, one scene, one N
# --------------------------------------------------------------------------- #


def fig_i(phys, touch, tri, splat, grid, n=4096.0):
    """Throughput of each tactile representation at the largest env count.

    This is the figure the optimisation is *for*. A kernel that removes a 4.6x
    overhead means nothing until the alternatives are priced; once they are, it
    is the difference between the hardware-faithful representation being
    unaffordable and costing about what the built-in one costs.

    Flex is absent by design -- see the note on GRID_CSV above. The touch grid is
    quoted at its own `nconmax`, which is marked on its bar, because it cannot be
    run at the reference budget at all; that is the finding of J, not a confound
    hidden in this chart.
    """
    rows = [
        ("physics only\n(no tactile)", phys, BLUE, 48),
        ("native <touch>\n421 sensors", touch, SKY, 48),
        ("splat, Triton\n1104 outputs", tri, GREEN, 48),
        ("splat, dense torch\n1104 outputs", splat, ORANGE, 48),
        ("binary touch grid\n300 patches", grid, PURPLE, 256),
    ]
    labels, rates, colours, budgets = [], [], [], []
    for label, d, colour, budget in rows:
        rate = at_envs(d, n, "env_steps_per_sec")
        if np.isnan(rate):
            raise SystemExit(f"{label!r} has no N={n:,.0f} rung")
        labels.append(label)
        rates.append(rate)
        colours.append(colour)
        budgets.append(budget)

    rates = np.asarray(rates)
    base = rates[0]
    ypos = np.arange(len(rows))[::-1]      # first row at the top

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(ypos, rates, color=colours, height=0.62, zorder=3)

    for y, rate, budget in zip(ypos, rates, budgets):
        cost = base / rate
        tag = f"{rate:,.0f}   ({cost:.2f}× cost)" if cost > 1.005 else f"{rate:,.0f}"
        if budget != 48:
            tag += f"   nconmax={budget}"
        ax.text(rate + base * 0.015, y, tag, va="center", fontsize=9,
                color="black" if cost > 1.005 else GREY)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, base * 1.42)
    ax.set_xlabel("throughput (physics env-steps/s, higher is better)")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, axis="x", color=FAINT, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=GREY, labelcolor="black", left=False)

    # Cheapest and dearest of the tactile rows -- row 0 is the no-tactile
    # baseline and is not itself a way of sensing touch.
    costs = base / rates[1:]
    titles(
        ax,
        f"Sensing touch costs {costs.min():.2f}× to {costs.max():.2f}×, "
        f"depending on the representation",
        "same scene, timestep and solver; the grid needs a larger contact budget and is marked as such\n"
        "flex omitted: different body tree, so its rate is not a like-for-like bar (see report Table I)",
    )
    save(fig, "I_four_representations.png")


# --------------------------------------------------------------------------- #
# J. Step time is affine in the ALLOCATED contact budget
# --------------------------------------------------------------------------- #


def fig_j(bnone, bgrid):
    """`ms/step` against `nconmax`, at fixed N and fixed live contact count.

    The generalisable result. mujoco_warp dispatches over `d.naconmax` -- the
    allocation -- not over the live contact count, so every step pays for every
    slot whether or not a contact occupies it. Two series with the same slope and
    very different geometry is the evidence that this is about allocation alone.
    """
    series = [
        ("bare scene (71 geoms)", bnone, BLUE, "o"),
        ("touch grid, collisions off (371 geoms)", bgrid, ORANGE, "s"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    style_axes(ax, xlabel="allocated contact budget (nconmax per env)", logx=False)

    fits = []
    for label, d, colour, marker in series:
        x, y = d["nconmax_per_env"], d["ms_per_step"]
        a, b, r2 = fit_affine(x, y)
        fits.append((label, a, b, r2, d))
        xs = np.linspace(0, x.max() * 1.05, 100)
        ax.plot(xs, a + b * xs, color=colour, lw=1.1, ls=(0, (5, 4)), alpha=0.85,
                zorder=2)
        ax.plot(x, y, color=colour, lw=0, marker=marker, ms=6.5, zorder=3,
                label=f"{label}\n     {a:.1f} + {b:.4f}·nconmax   (R²={r2:.3f})")

    # The reference budget, and the one the grid is forced to.
    for xv, note in ((48, "reference\nnconmax=48"), (256, "grid needs\n≥256")):
        ax.axvline(xv, color=GREY, lw=0.9, ls=(0, (2, 3)), zorder=1)
        ax.text(xv + 4, ax.get_ylim()[1] * 0.02, note, fontsize=8, color=GREY,
                va="bottom", ha="left")

    ax.set_xlim(0, max(d["nconmax_per_env"].max() for _, d in
                       [(s[0], s[1]) for s in series]) * 1.06)
    ax.set_ylim(0, max(d["ms_per_step"].max() for _, d in
                       [(s[0], s[1]) for s in series]) * 1.15)
    ax.set_ylabel("wall-clock per step (ms)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, borderpad=0.2,
              labelspacing=0.9)

    occ = np.concatenate([d["contacts_per_env"] for _, _, _, _, d in fits])
    slopes = [b for _, _, b, _, _ in fits]
    titles(
        ax,
        "Step cost is linear in the budget allocated, not the contacts used",
        f"N = 1,024 fixed · njmax = 120 fixed · live contacts pinned at "
        f"{np.nanmin(occ):.1f}–{np.nanmax(occ):.1f} per env across every point\n"
        f"the two slopes agree to {100 * abs(slopes[0] - slopes[1]) / np.mean(slopes):.1f}%, "
        f"so the cost is the allocation and not the geometry",
    )
    save(fig, "J_contact_budget.png")


def main() -> int:
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.handlelength": 2.4,
    })

    phys = load(PHYSICS_CSV)
    splat = load(SPLAT_CSV)
    tri = load(TRITON_CSV)
    kern = load(KERNEL_CSV)
    touch = load(TOUCH_CSV)
    grid = load(GRID_CSV)
    bnone = load_budget(BUDGET_NONE_CSV)
    bgrid = load_budget(BUDGET_GRID_CSV)

    fig_a(phys, splat)
    fig_b(phys, splat)
    fig_c(phys, splat)
    fig_d(phys, splat)
    fig_e(phys, splat, tri)
    fig_f(kern)
    fig_g(splat, tri, kern)
    fig_h(phys, splat, tri, kern)
    fig_i(phys, touch, tri, splat, grid)
    fig_j(bnone, bgrid)

    # Echo the derived headline numbers so a figure can be checked against the
    # CSVs without opening a PNG.
    common, (ip, isp, it) = align(phys, splat, tri)
    slow = phys["env_steps_per_sec"][ip] / splat["env_steps_per_sec"][isp]
    slow_tri = phys["env_steps_per_sec"][ip] / tri["env_steps_per_sec"][it]
    e2e = tri["env_steps_per_sec"][it] / splat["env_steps_per_sec"][isp]
    frac = float(splat["encoder_frac"][-1])
    ceiling = 1.0 / (1.0 - frac)
    k = int(np.argmax(kern["num_envs"]))
    stage = float(tri["encoder_ms_per_step"][it][-1])
    kernel = float(kern["triton_ms"][k])

    print(f"\ncrossover N (torch)       = {crossover_n(splat):,.0f}")
    print(f"linear knee (physics)     = {linear_knee(phys):,.0f}")
    print(f"encoder share at N={common[-1]:,.0f}  = {100 * frac:.1f}%  (torch)")
    print(f"                          = {100 * tri['encoder_frac'][it][-1]:.1f}%  (Triton)")
    print(f"Amdahl ceiling            = {ceiling:.2f}x")
    print(f"cost of touch at N={common[-1]:,.0f} = {slow[-1]:.2f}x -> {slow_tri[-1]:.2f}x")
    print(f"end-to-end gain           = {e2e[-1]:.2f}x  "
          f"({100 * e2e[-1] / ceiling:.0f}% of the ceiling)")
    print(f"kernel at N={kern['num_envs'][k]:,.0f}         = "
          f"{kern['triton_ms'][k]:.3f} ms  "
          f"({kern['speedup_vs_eager'][k]:.0f}x eager, "
          f"{kern['speedup_vs_compile'][k]:.0f}x torch.compile)")
    print(f"torch.compile alone       = {kern['compile_speedup_vs_eager'][k]:.2f}x  "
          f"(-> {amdahl(frac, kern['compile_speedup_vs_eager'][k]):.2f}x end to end)")
    print(f"Amdahl: 30x kernel        = {amdahl(frac, 30.0):.2f}x  vs "
          f"{amdahl(frac, kern['speedup_vs_eager'][k]):.2f}x at "
          f"{kern['speedup_vs_eager'][k]:.0f}x")
    print(f"tactile stage vs kernel   = {stage:.3f} ms vs {kernel:.3f} ms  "
          f"({100 * (1 - kernel / stage):.0f}% pre-processing, inferred)")

    nmax = common[-1]
    base = at_envs(phys, nmax, "env_steps_per_sec")
    print(f"\ncost of sensing at N={nmax:,.0f}:")
    for name, d, budget in (("native <touch>", touch, 48),
                            ("splat, Triton", tri, 48),
                            ("splat, dense", splat, 48),
                            ("touch grid", grid, 256)):
        rate = at_envs(d, nmax, "env_steps_per_sec")
        print(f"  {name:<16} {rate:>9,.0f} env-steps/s  {base / rate:>5.2f}x"
              + (f"   (nconmax={budget})" if budget != 48 else ""))
    for label, d in (("bare scene", bnone), ("grid, collide off", bgrid)):
        a, b, r2 = fit_affine(d["nconmax_per_env"], d["ms_per_step"])
        occ = d["contacts_per_env"]
        print(f"budget law, {label:<18} ms/step = {a:.2f} + {b:.4f} x nconmax  "
              f"(R^2={r2:.4f}, occupancy {np.nanmin(occ):.1f}-{np.nanmax(occ):.1f}/env)")

    _mirror_to_figures()
    return 0


def _mirror_to_figures() -> None:
    """Copy the generated PNGs to figures/ for the LaTeX report, atomically.

    Each file is written under a temporary name in the *same* directory and then
    `os.replace`d over the target, which is atomic on POSIX. `shutil.copy2`
    straight onto the destination is not: it truncates the target and then
    streams into it, so anything reading concurrently sees a partial file.

    That is not hypothetical here. Rebuilding the figures while a LaTeX run is in
    flight -- the obvious thing to do when iterating on a plot -- makes pdflatex
    read a half-written PNG and fail with

        pdfTeX error: pdflatex (file ./figures/X.png): reading image file failed

    which looks like a corrupt figure rather than a race, and is gone by the time
    you inspect the file. Same technique, and same reasoning, as
    `scale_sweep.py:flush_csv`.
    """
    import shutil

    FIGURES_DIR.mkdir(exist_ok=True)
    pngs = sorted(OUT.glob("*.png"))
    for png in pngs:
        dest = FIGURES_DIR / png.name
        tmp = dest.with_suffix(".png.tmp")
        shutil.copy2(png, tmp)
        os.replace(tmp, dest)
    print(f"mirrored {len(pngs)} figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
