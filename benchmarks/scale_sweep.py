"""Ramp the env count, one memory-capped child process per N, and find the ceiling.

Why subprocesses: this machine has unified CPU/GPU memory, so an oversized
allocation exhausts system RAM and wedges the host rather than raising
`CUDA out of memory` (PROJECT_LOG.md §1.1 -- it already took the box down once).
Each measurement therefore runs in its own `systemd-run --scope` with a hard
`MemoryMax`, so the kernel kills a child instead of the machine. A failure at
some N is a *result* -- the memory ceiling -- not a crash.

Ramps until one of: the list is exhausted, a child dies, or the projected
footprint would exceed the cap.

    .venv/bin/python benchmarks/scale_sweep.py
    .venv/bin/python benchmarks/scale_sweep.py --max-envs 2048 --mem-cap 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS = ROOT / "benchmarks" / "harness.py"
RESULTS = ROOT / "benchmarks" / "results"
PYTHON = ROOT / ".venv" / "bin" / "python"

DEFAULT_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

FIELDS = [
    "num_envs", "tactile", "ok", "env_steps_per_sec", "us_per_env_step",
    "ms_per_step", "peak_rss_gb", "rss_after_alloc_gb", "gb_per_env_mb",
    "setup_s", "warmup_s", "wall_s", "steps", "device",
    "sim_dt", "contacts", "contacts_per_env", "overflow_worlds", "naconmax_total", "njmax_actual",
    "physics_ms_per_step", "encoder_ms_per_step", "encoder_frac", "torch_peak_gb",
    "dropped_contacts", "active_taxels_world0",
    # --tactile touch only. The step/sensor split is a difference of fused
    # loops rather than a stopwatch around a stage (harness.py docstring), so
    # the components are carried through to the CSV: without them the reader
    # cannot tell the 421-sensor dispatch from mujoco_warp's fixed sensor-stage
    # overhead, and `timing_noise_ms_per_step` is what says whether either
    # difference is above the noise at that N.
    "n_touch_sensors", "touch_scene", "touch_kernel_ms_per_step",
    "sensor_stage_ms_per_step", "readout_ms_per_step",
    "step_all_on_ms_per_step", "step_touch_off_ms_per_step",
    "step_sensors_off_ms_per_step", "timing_noise_ms_per_step",
    "touch_active_world0", "touch_active_any_world", "touch_active_mean_per_world",
    "touch_min_nonzero_n", "touch_max_n",
    # --tactile touchgrid only. `ngeom` and `touchgrid_collide` identify which
    # of the three variants a row is, and the contact breakdown is what backs
    # the H3 claim: `touchgrid_patch_cube_contacts` against
    # `touchgrid_patch_patch_contacts` is the difference between "the grid is
    # expensive because it senses the object" and "the grid is expensive
    # because it fights itself". `overflow_worlds` above is the other half.
    "ngeom", "touchgrid_collide", "n_patches", "step_only_ms_per_step",
    "touchgrid_active_world0", "touchgrid_active_any_world",
    "touchgrid_active_mean_per_world", "touchgrid_contacts_live",
    "touchgrid_patch_contacts", "touchgrid_patch_patch_contacts",
    "touchgrid_patch_cube_contacts", "touchgrid_patch_floor_contacts",
    "touchgrid_patch_hand_contacts",
    # --motion grasp only. With the supervisor's grasp patterns driving the
    # hand the contact count is a distribution, not a constant, so the single
    # `contacts` snapshot above stops being a description of the run.
    # `world_contacts_min`/`_max` is the per-world spread at an instant -- the
    # load imbalance a gather kernel actually feels, and the thing the frozen
    # pose could not produce at all.
    "motion", "contacts_min", "contacts_max", "contacts_mean",
    "contacts_per_env_min", "contacts_per_env_max", "contacts_per_env_mean",
    "world_contacts_min", "world_contacts_max", "contact_sample_steps",
    "note",
]


def run_one(n: int, steps: int, warmup: int, mem_cap_gb: int, timeout_s: int,
            tactile: str = "none", encoder: str = "torch",
            touch_scene: str = "inject", touchgrid_collide: str = "object",
            nconmax: int = 48, njmax: int = 120,
            motion: str = "static") -> dict:
    """One measurement, in its own memory-capped scope."""
    cmd = []
    if shutil.which("systemd-run"):
        cmd += [
            "systemd-run", "--user", "--scope", "-q",
            "-p", f"MemoryMax={mem_cap_gb}G",
            "-p", "MemorySwapMax=0",
        ]
    cmd += [str(PYTHON), str(HARNESS), "--num-envs", str(n),
            "--steps", str(steps), "--warmup", str(warmup),
            "--tactile", tactile, "--encoder", encoder,
            "--touch-scene", touch_scene,
            "--touchgrid-collide", touchgrid_collide,
            "--nconmax-per-env", str(nconmax), "--njmax-per-env", str(njmax),
            "--motion", motion]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return dict(num_envs=n, ok=False, note=f"timeout >{timeout_s}s")
    took = time.perf_counter() - t0

    # harness prints exactly one JSON object on stdout
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break

    tail = (proc.stderr or "").strip().splitlines()
    note = tail[-1][:120] if tail else f"rc={proc.returncode}"
    if proc.returncode == -9 or "MemoryMax" in note or "Killed" in note:
        note = f"OOM-killed at cap {mem_cap_gb}G"
    return dict(num_envs=n, ok=False, note=note, wall_s=took)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--mem-cap", type=int, default=60,
                    help="hard MemoryMax per child, GB. Keep well under total RAM.")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tag", default="physics_only")
    ap.add_argument("--tactile", choices=["none", "splat", "touch", "touchgrid"],
                    default="none")
    ap.add_argument("--encoder", choices=["torch", "compile", "triton"], default="torch")
    ap.add_argument("--touch-scene", choices=["inject", "supervisor"], default="inject")
    ap.add_argument("--touchgrid-collide", choices=["object", "naive", "off"],
                    default="object")
    # The supervisor's budget is the default and is what every banked sweep used.
    # The touchgrid does not fit in it -- raising these is a deliberate,
    # reportable act, which is why they are explicit flags and not auto-sized.
    ap.add_argument("--nconmax-per-env", type=int, default=48)
    ap.add_argument("--njmax-per-env", type=int, default=120)
    ap.add_argument("--motion", choices=["static", "grasp"], default="static")
    args = ap.parse_args()

    ladder = [n for n in DEFAULT_LADDER if n <= args.max_envs]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"scale_sweep_{args.tag}.csv"

    print(f"ramping N={ladder}")
    print(f"cap {args.mem_cap}G per child, {args.steps} timed steps -> {out_csv}\n")
    print(f"{'N':>6}{'env-steps/s':>15}{'us/env-step':>13}{'ms/step':>10}"
          f"{'peak GB':>10}{'con/env':>9}{'ovf':>5}  status")
    print("-" * 74)

    def flush_csv(rows: list) -> None:
        """Rewrite the CSV from scratch after every rung.

        The sweep used to write once, at the end. That is fine right up until it
        isn't: a rung that wedges the box, an OOM in the parent, a dropped SSH,
        or a Ctrl-C at rung 12 threw away all twelve completed measurements, each
        of which cost real GPU time. This box has form -- unified memory means an
        oversized allocation takes down the host, not just the process
        (PROJECT_LOG §1.1).

        Rewriting the whole file each time rather than appending keeps the header
        correct and costs nothing at 13 rows. The write goes to a temp file and
        is renamed over the target, so a crash *during* the write cannot leave a
        half-written CSV where a complete one used to be.
        """
        tmp = out_csv.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        os.replace(tmp, out_csv)

    rows = []
    for n in ladder:
        r = run_one(n, args.steps, args.warmup, args.mem_cap, args.timeout,
                    tactile=args.tactile, encoder=args.encoder,
                    touch_scene=args.touch_scene,
                    touchgrid_collide=args.touchgrid_collide,
                    nconmax=args.nconmax_per_env, njmax=args.njmax_per_env,
                    motion=args.motion)
        r.setdefault("tactile", args.tag)
        r.setdefault("note", "")
        if r.get("ok"):
            # crude but useful: marginal footprint over the N=1 baseline
            base = rows[0].get("peak_rss_gb") if rows and rows[0].get("ok") else None
            r["gb_per_env_mb"] = (
                (r["peak_rss_gb"] - base) * 1024.0 / n if base and n else float("nan")
            )
            print(f"{n:>6}{r['env_steps_per_sec']:>15,.0f}{r['us_per_env_step']:>13.1f}"
                  f"{r['ms_per_step']:>10.2f}{r['peak_rss_gb']:>10.2f}"
                  f"{r.get('contacts_per_env', float('nan')):>9.1f}"
                  f"{r.get('overflow_worlds', 0):>5}  ok")
        else:
            print(f"{n:>6}{'-':>15}{'-':>13}{'-':>10}{'-':>10}{'-':>9}  "
                  f"FAILED: {r['note']}")
        rows.append(r)
        flush_csv(rows)          # banked before the next rung can take the box down
        sys.stdout.flush()       # ... and make the log live, not block-buffered
        if not r.get("ok"):
            print(f"\nceiling reached at N={n}. Stopping.")
            break

    flush_csv(rows)

    ok = [r for r in rows if r.get("ok")]
    print(f"\nwrote {out_csv}  ({len(ok)}/{len(rows)} succeeded)")
    if len(ok) >= 2:
        a, b = ok[0], ok[-1]
        print(f"per-env cost {a['us_per_env_step']:.1f} us at N={a['num_envs']} "
              f"-> {b['us_per_env_step']:.1f} us at N={b['num_envs']}  "
              f"({a['us_per_env_step'] / b['us_per_env_step']:.1f}x better)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
