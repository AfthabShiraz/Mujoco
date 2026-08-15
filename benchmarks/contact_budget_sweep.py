"""Scan the ALLOCATED contact budget at a fixed env count, and fit the cost law.

WHY THIS EXISTS. `HYPOTHESES.md` H3 found the result most likely to outlive this
project: mujoco_warp sizes its kernel launches off `d.naconmax` -- the contact
budget you *allocated* -- and not off the live contact count. So step time is an
affine function of `nconmax` and is blind to how much of it is occupied. At
N=1024 the difference between `nconmax=48` and `nconmax=256` is ~3.4x end to end
with identical physics.

That was established with a handful of one-off runs whose numbers lived only in
the hypothesis register. This script turns it into a banked measurement: one
memory-capped child per budget, a CSV, and a least-squares fit printed at the
end. `analysis/plots.py` plots the CSV; nothing about the figure is hardcoded.

THE ISOLATION IS THE POINT. Every row holds the scene, the env count, the
timestep, the solver and `njmax` fixed, and the pose is the frozen 60%-closed
grasp, so the number of *live* contacts is pinned across the whole scan. The only
thing that varies is how much contact space was reserved. Two series are worth
banking:

  --tactile none                       the bare scene. The general claim: this is
                                       a property of mujoco_warp, not of tactile
                                       sensing or of any model we injected.
  --tactile touchgrid                  the 300-patch grid with its collisions
    --touchgrid-collide off            switched OFF, so the patches are present
                                       and inert. This is H3's own isolation --
                                       the geometry that forces a big budget is
                                       in the scene, but generates no contacts,
                                       which is what separates "the contacts cost"
                                       from "the allocation costs".

Both should give the same slope. That they do is the evidence that the law is
about allocation and nothing else.

SAFETY. Same rule as every other sweep here: one memory-capped `systemd-run`
scope per point, because this box has unified CPU/GPU memory and an oversized
allocation wedges the host rather than raising `CUDA out of memory`
(PROJECT_LOG.md 1.1). A large `nconmax` is exactly the knob that would do it --
`naconmax` is a TOTAL across worlds, so 384 slots/env at N=1024 reserves 393,216
contacts. Ramp, do not jump.

    .venv/bin/python benchmarks/contact_budget_sweep.py --tag none_n1024
    .venv/bin/python benchmarks/contact_budget_sweep.py \
        --tactile touchgrid --touchgrid-collide off --tag grid_off_n1024
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from scale_sweep import FIELDS, run_one  # noqa: E402  (path set above)

RESULTS = ROOT / "benchmarks" / "results"

# H3's original scan points, kept so the banked CSV is comparable to the numbers
# already quoted in the hypothesis register and the report.
DEFAULT_BUDGETS = [48, 96, 128, 192, 256, 384]

# `nconmax_per_env` is not in scale_sweep's FIELDS (it is a constant there, one
# value per sweep). Here it is the independent variable, so it leads the row.
BUDGET_FIELDS = ["nconmax_per_env"] + [f for f in FIELDS if f != "nconmax_per_env"]


def fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares `y = a + b*x`, returning (a, b, R^2).

    Hand-rolled rather than pulled from numpy because this file otherwise needs
    nothing beyond the standard library, and six points do not justify the import.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1024,
                    help="held fixed across the scan; the budget is what varies")
    ap.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    ap.add_argument("--njmax-per-env", type=int, default=120,
                    help="held fixed. H3 measured njmax as costing nothing; "
                         "raising it here would confound the scan.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--mem-cap", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tactile", choices=["none", "splat", "touch", "touchgrid"],
                    default="none")
    ap.add_argument("--encoder", choices=["torch", "compile", "triton"],
                    default="torch")
    ap.add_argument("--touchgrid-collide", choices=["object", "naive", "off"],
                    default="off")
    ap.add_argument("--motion", choices=["static", "grasp"], default="static")
    ap.add_argument("--tag", default="none_n1024")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"contact_budget_{args.tag}.csv"

    print(f"N={args.num_envs} fixed, njmax={args.njmax_per_env}/env fixed, "
          f"tactile={args.tactile}"
          + (f" collide={args.touchgrid_collide}" if args.tactile == "touchgrid" else ""))
    print(f"scanning nconmax/env = {args.budgets} -> {out_csv}\n")
    print(f"{'nconmax':>8}{'naconmax':>11}{'ms/step':>10}{'env-steps/s':>14}"
          f"{'con/env':>9}{'ovf':>5}  status")
    print("-" * 62)

    def flush_csv(rows: list) -> None:
        """Rewrite after every point, same reasoning as scale_sweep.flush_csv:
        a wedged box at point 5 must not throw away points 1-4."""
        tmp = out_csv.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=BUDGET_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        os.replace(tmp, out_csv)

    rows: list[dict] = []
    for nc in args.budgets:
        r = run_one(args.num_envs, args.steps, args.warmup, args.mem_cap,
                    args.timeout, tactile=args.tactile, encoder=args.encoder,
                    touchgrid_collide=args.touchgrid_collide,
                    nconmax=nc, njmax=args.njmax_per_env, motion=args.motion)
        r["nconmax_per_env"] = nc
        r.setdefault("tactile", args.tactile)
        r.setdefault("note", "")
        if r.get("ok"):
            print(f"{nc:>8}{r.get('naconmax_total', 0):>11,}"
                  f"{r['ms_per_step']:>10.2f}{r['env_steps_per_sec']:>14,.0f}"
                  f"{r.get('contacts_per_env', float('nan')):>9.1f}"
                  f"{r.get('overflow_worlds', 0):>5}  ok")
        else:
            print(f"{nc:>8}{'-':>11}{'-':>10}{'-':>14}{'-':>9}{'-':>5}  "
                  f"FAILED: {r['note']}")
        rows.append(r)
        flush_csv(rows)
        sys.stdout.flush()

    ok = [r for r in rows if r.get("ok")]
    print(f"\nwrote {out_csv}  ({len(ok)}/{len(rows)} succeeded)")

    if len(ok) >= 3:
        xs = [float(r["nconmax_per_env"]) for r in ok]
        ys = [float(r["ms_per_step"]) for r in ok]
        a, b, r2 = fit_affine(xs, ys)
        print(f"\nfit: ms/step = {a:.2f} + {b:.4f} x nconmax     R^2 = {r2:.4f}")
        lo, hi = ok[0], ok[-1]
        print(f"throughput {lo['env_steps_per_sec']:,.0f} -> {hi['env_steps_per_sec']:,.0f} "
              f"env-steps/s over nconmax {lo['nconmax_per_env']} -> {hi['nconmax_per_env']} "
              f"({lo['env_steps_per_sec'] / hi['env_steps_per_sec']:.2f}x)")
        con = [r.get("contacts_per_env") for r in ok if r.get("contacts_per_env")]
        if con:
            print(f"live contacts/env across the scan: "
                  f"{min(con):.1f} to {max(con):.1f}  <- occupancy is pinned; "
                  f"only the allocation moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
