"""Regenerate every figure in the write-up from the banked benchmark CSVs.

Nothing here is hardcoded from a previous run: every number that appears on a
figure -- the crossover env count, the linear-scaling knee, the slowdown factor,
the Amdahl ceiling -- is derived from
`benchmarks/results/scale_sweep_physics_only.csv` and
`benchmarks/results/scale_sweep_splat.csv` at plot time. Re-run the sweeps and
the figures (and their titles) follow. That is deliberate: a figure whose
caption drifts from its data is worse than no figure.

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
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless box: no display, no interactive backend

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "analysis" / "plots"
DPI = 150

PHYSICS_CSV = RESULTS / "scale_sweep_physics_only.csv"
SPLAT_CSV = RESULTS / "scale_sweep_splat.csv"

# Okabe-Ito. BLUE/ORANGE are the two that must never be confused (they carry the
# crossover), and they are the best-separated pair in the set under CVD.
BLUE = "#0072B2"    # physics
ORANGE = "#E69F00"  # encoder / tactile path
GREEN = "#009E73"   # slowdown
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
    A failed rung -- the memory ceiling -- is a legitimate row in these files and
    must not be plotted as a data point, hence the `ok` filter.
    """
    with path.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["ok"].strip().lower() == "true"]
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


def matched(a: dict[str, np.ndarray], b: dict[str, np.ndarray]):
    """Indices into `a` and `b` for the env counts both sweeps completed."""
    common = np.intersect1d(a["num_envs"], b["num_envs"])
    ia = np.searchsorted(a["num_envs"], common)
    ib = np.searchsorted(b["num_envs"], common)
    return common, ia, ib


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


def style_axes(ax, xlabel="environments (N)", logx=True, logy=False):
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([f"{t:,}" for t in XTICKS])
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
            f"~{contacts:.1f} contacts/env · contact overflow: {overflow}")


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

    ax.plot(common, slowdown, color=GREEN, lw=2.2, marker="D", ms=5.0)
    ax.annotate(f"{slowdown[i_max]:.2f}×", xy=(common[i_max], slowdown[i_max]),
                xytext=(-4, 8), textcoords="offset points",
                fontsize=10, color=GREEN, fontweight="bold", ha="right")

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

    fig_a(phys, splat)
    fig_b(phys, splat)
    fig_c(phys, splat)
    fig_d(phys, splat)

    # Echo the derived headline numbers so a figure can be checked against the
    # CSVs without opening a PNG.
    common, ip, isp = matched(phys, splat)
    slow = phys["env_steps_per_sec"][ip] / splat["env_steps_per_sec"][isp]
    top = 100.0 * splat["encoder_frac"][-1]
    print(f"\ncrossover N            = {crossover_n(splat):,.0f}")
    print(f"linear knee (physics)  = {linear_knee(phys):,.0f}")
    print(f"encoder share at N={splat['num_envs'][-1]:,.0f} = {top:.1f}%")
    print(f"Amdahl ceiling         = {1.0 / (1.0 - top / 100.0):.2f}x")
    print(f"slowdown at N={common[-1]:,.0f}     = {slow[-1]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
