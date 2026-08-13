"""Measure MuJoCo Warp throughput and memory for ONE env count, then exit.

Deliberately single-shot: it measures one N and terminates. `scale_sweep.py`
runs it once per N in a separate memory-capped process, so an out-of-memory at
high N kills a child instead of wedging the machine (PROJECT_LOG.md §1.1 --
this box has unified memory, so an oversized run takes down the host, not just
the process).

Timing rules, because getting these wrong makes the numbers meaningless:
  * warm up first -- kernel compilation and allocation dominate the first steps
  * `wp.synchronize()` before starting and before stopping the clock; GPU work
    is asynchronous, so timing without it measures queue-submission speed
  * report env-steps/sec (N x steps / wall), the scale-invariant number

Emits one JSON object on stdout. Any human-readable commentary goes to stderr.

Run directly (prefer scale_sweep.py):
    .venv/bin/python benchmarks/harness.py --num-envs 64
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCENE = (
    ROOT / "third_party" / "leapXelaMjLab" / "src" / "leap_xela_mjlab"
    / "assets" / "leapXELA_model" / "scene_mjx_cube.xml"
)

# Matches leapXelaMjLab env_cfg.py; the static contact budget is part of what
# we are measuring, so it must not drift.
#
# CAREFUL -- these two are scoped differently in mujoco_warp's put_data, and
# getting it wrong silently destroys the measurement:
#   naconmax  is a TOTAL across worlds  -> multiply by nworld (default: 48*nworld)
#   njmax     is PER WORLD              -> do NOT multiply (default: 64)
# Passing njmax=120*nworld makes each world's constraint space nworld times too
# large; the dense solver (JTDAJ + Cholesky) then costs O(N^2) and throughput
# *falls* as N rises. Measured: N=512 went 337 -> 67,840 env-steps/s on fixing it.
NCONMAX_PER_ENV = 48
NJMAX_PER_ENV = 120

# The hand must actually be gripping, or the benchmark times a contact-free
# scene and tells you nothing about contact cost.
CLOSE_FRACTION = 0.6
SETTLE_STEPS = 250   # CPU run shows first cube contact ~step 120


def peak_rss_gb() -> float:
    """Peak resident set of this process. On unified memory this tracks the
    GPU allocation too, which is exactly the number we care about here."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--scene", type=pathlib.Path, default=DEFAULT_SCENE)
    ap.add_argument("--tactile", choices=["none"], default="none",
                    help="representation under test; only the H1 control exists so far")
    args = ap.parse_args()

    t_import = time.perf_counter()
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING   # module-load chatter would drown the JSON
    import mujoco_warp as mjw
    import_s = time.perf_counter() - t_import

    n = args.num_envs
    print(f"[harness] N={n} scene={args.scene.name}", file=sys.stderr)

    mjm = mj.MjModel.from_xml_path(str(args.scene))
    mjd = mj.MjData(mjm)
    mj.mj_forward(mjm, mjd)

    t0 = time.perf_counter()
    m = mjw.put_model(mjm)
    d = mjw.put_data(
        mjm, mjd, nworld=n,
        naconmax=NCONMAX_PER_ENV * n,   # total across worlds
        njmax=NJMAX_PER_ENV,            # per world -- do not scale
    )
    wp.synchronize()
    setup_s = time.perf_counter() - t0
    rss_after_alloc = peak_rss_gb()

    # ---- close the hand so contacts exist ----------------------------------
    lo = mjm.actuator_ctrlrange[:, 0]
    hi = mjm.actuator_ctrlrange[:, 1]
    target = lo + CLOSE_FRACTION * (hi - lo)
    d.ctrl.assign(np.tile(target.astype(np.float32), (n, 1)))
    for _ in range(SETTLE_STEPS):
        mjw.step(m, d)
    wp.synchronize()

    # ---- warm up: first steps pay for kernel codegen, not physics ----------
    t0 = time.perf_counter()
    for _ in range(args.warmup):
        mjw.step(m, d)
    wp.synchronize()
    warmup_s = time.perf_counter() - t0

    contacts = int(d.nacon.numpy()[0])
    overflow = int((d.overflow.numpy() != 0).sum())

    # ---- timed ------------------------------------------------------------
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        mjw.step(m, d)
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    env_steps = n * args.steps
    result = dict(
        num_envs=n,
        tactile=args.tactile,
        steps=args.steps,
        warmup=args.warmup,
        scene=args.scene.name,
        nconmax_per_env=NCONMAX_PER_ENV,
        njmax_per_env=NJMAX_PER_ENV,
        naconmax_total=int(d.naconmax),
        njmax_actual=int(d.njmax),
        contacts=contacts,
        contacts_per_env=contacts / n,
        overflow_worlds=overflow,
        wall_s=elapsed,
        ms_per_step=1000.0 * elapsed / args.steps,
        env_steps_per_sec=env_steps / elapsed,
        us_per_env_step=1e6 * elapsed / env_steps,
        setup_s=setup_s,
        warmup_s=warmup_s,
        import_s=import_s,
        peak_rss_gb=peak_rss_gb(),
        rss_after_alloc_gb=rss_after_alloc,
        device=str(wp.get_device()),
        ok=True,
    )
    print(json.dumps(result))
    print(f"[harness] {result['env_steps_per_sec']:,.0f} env-steps/s  "
          f"{result['ms_per_step']:.2f} ms/step  peak {result['peak_rss_gb']:.2f} GB  "
          f"contacts {contacts} ({contacts / n:.1f}/env)  overflow {overflow}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
