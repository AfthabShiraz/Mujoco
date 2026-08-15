#!/usr/bin/env python
"""H4: does the flex/bubble tactile skin run on MuJoCo Warp, and at what cost?

Deliberately a separate script from `harness.py`, and the reason is worth stating
rather than leaving as a code-layout accident: **the flex hand is not the hand
the other three representations were measured on.** It lives in the tactile
lineage (`leapXELA_model/scene_flex_sensor_Box.xml`, 387 bodies / 1120 joints /
18 flexes / 386 equality constraints) rather than the pinned rigid RL model that
`scene_mjx_cube.xml` loads (20 bodies). There is no injection that turns one into
the other -- a flex skin is not a site or a geom you can add to a rigid link, it
is a different body tree. Running it through the same harness would put a number
in the same table as the splat/touch/touchgrid rows and invite a comparison that
the models do not support.

So this measures the one thing that IS comparable, which is also exactly what H4
asks: the **order of magnitude** of a flex step against a rigid step, and whether
the thing survives being replicated across worlds at all.

Reported honestly, that means:
  * `ms_per_step` here is NOT directly comparable to the rigid sweeps' ms/step,
    because the two run at different timesteps (0.002 vs 0.01) and a step is
    therefore a different amount of simulated time. Use
    `equiv_env_steps_per_sec`, which rescales by `sim_dt / RIGID_DT`.
  * Even that is not a like-for-like fourth bar for the cost-of-tactile-sensing
    figure: it is a different hand in a different model lineage. It is an
    order-of-magnitude result with its own caveat, not a row in the table.

`njmax` is per world and the flex model needs >= 3030 of them (mujoco_warp says
so explicitly, and cleanly, at `put_data` -- a nice contrast with the touchgrid,
which overruns its contact budget by dying inside `ccd_kernel`).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import resource
import sys
import time

import mujoco as mj
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLEX_SCENE = ROOT / "third_party" / "leapXELA_model" / "scene_flex_sensor_Box.xml"
RESULTS = ROOT / "benchmarks" / "results"

# The rigid sweeps run at 0.01 s, matching leapXelaMjLab's env_cfg. The FLEX
# model cannot: it declares timestep=0.002 with 50 solver iterations, and forcing
# 0.01 on 1120 deformable DoF makes it diverge -- MuJoCo prints
# "Nan, Inf or huge value in QACC ... the simulation is unstable" and the
# Kelvin-Voigt readout returns forces of order 1e5 N.
#
# An earlier version of this script did exactly that, so the first flex numbers
# reported for H4 were taken on a misconfigured and intermittently diverging
# simulation. They are superseded.
#
# The model's own timestep is used instead, and throughput is additionally
# reported at MATCHED SIMULATED TIME: a 0.002 s step advances one fifth as far as
# a 0.01 s step, so raw steps/s overstates flex by 5x against the rigid sweeps.
# `equiv_env_steps_per_sec` is the number that may be compared to them.
RIGID_DT = 0.01         # what the rigid sweeps and env_cfg.py use
CLOSE_FRACTION = 0.6    # matches harness.py, so the grasp is the same gesture
# Settle is specified in SIMULATED SECONDS, not steps. A fixed step count is a
# trap once the timestep is not 0.01: 60 steps is 0.6 s at dt=0.01 but only
# 0.12 s at the flex model's 0.002, which is not long enough for the hand to
# close -- the first corrected run reported 0 contacts for exactly this reason.
SETTLE_S = 0.6


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--njmax", type=int, default=4096)
    ap.add_argument("--naconmax-per-env", type=int, default=48)
    ap.add_argument("--scene", type=pathlib.Path, default=FLEX_SCENE)
    ap.add_argument("--no-grip", action="store_true",
                    help="skip the closing gesture -- the contact-free floor")
    args = ap.parse_args()

    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    import mujoco_warp as mjw

    n = args.num_envs
    m0 = mj.MjModel.from_xml_path(str(args.scene))
    # Deliberately NOT overridden -- see the note on RIGID_DT above.
    sim_dt = float(m0.opt.timestep)
    d0 = mj.MjData(m0)
    mj.mj_forward(m0, d0)

    t0 = time.perf_counter()
    m = mjw.put_model(m0)
    d = mjw.put_data(m0, d0, nworld=n,
                     naconmax=args.naconmax_per_env * n, njmax=args.njmax)
    wp.synchronize()
    setup_s = time.perf_counter() - t0

    # Close the hand, same gesture as the rigid harness, so "gripping" means the
    # same thing in both. Without this the flex skin never touches anything and
    # the number is the contact-free floor rather than the cost under load.
    if not args.no_grip:
        lo, hi = m0.actuator_ctrlrange[:, 0], m0.actuator_ctrlrange[:, 1]
        target = lo + CLOSE_FRACTION * (hi - lo)
        d.ctrl.assign(np.tile(target.astype(np.float32), (n, 1)))
        for _ in range(int(round(SETTLE_S / sim_dt))):
            mjw.step(m, d)
        wp.synchronize()

    for _ in range(args.warmup):
        mjw.step(m, d)
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        mjw.step(m, d)
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    nacon = int(d.nacon.numpy()[0])
    overflow = int((d.overflow.numpy() != 0).sum())
    result = dict(
        num_envs=n, tactile="flex", ok=True,
        scene=args.scene.name,
        nbody=int(m0.nbody), njnt=int(m0.njnt), nflex=int(m0.nflex),
        neq=int(m0.neq), nv=int(m0.nv),
        sim_dt=sim_dt, rigid_dt=RIGID_DT, steps=args.steps,
        iterations=int(m0.opt.iterations), ls_iterations=int(m0.opt.ls_iterations),
        gripping=not args.no_grip,
        ms_per_step=1000.0 * elapsed / args.steps,
        env_steps_per_sec=n * args.steps / elapsed,
        # Comparable to the rigid sweeps: scales raw steps by how much simulated
        # time each one actually advances.
        equiv_env_steps_per_sec=(n * args.steps / elapsed) * (sim_dt / RIGID_DT),
        us_per_env_step=1e6 * elapsed / (n * args.steps),
        contacts=nacon, contacts_per_env=nacon / n,
        overflow_worlds=overflow,
        naconmax_total=int(d.naconmax), njmax_actual=int(d.njmax),
        setup_s=setup_s, wall_s=elapsed, peak_rss_gb=peak_rss_gb(),
        device=str(wp.get_device()),
    )
    print(json.dumps(result))
    print(f"[flex] N={n} dt={sim_dt} it={int(m0.opt.iterations)}  "
          f"{result['ms_per_step']:.2f} ms/step  "
          f"{result['env_steps_per_sec']:,.0f} raw steps/s  "
          f"{result['equiv_env_steps_per_sec']:,.0f} equiv-env-steps/s  "
          f"peak {result['peak_rss_gb']:.2f} GB  "
          f"contacts {nacon} ({nacon / n:.1f}/env)  overflow {overflow}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
