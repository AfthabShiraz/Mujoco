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
    "contacts", "contacts_per_env", "overflow_worlds", "naconmax_total", "njmax_actual",
    "physics_ms_per_step", "encoder_ms_per_step", "encoder_frac", "torch_peak_gb",
    "dropped_contacts", "active_taxels_world0", "note",
]


def run_one(n: int, steps: int, warmup: int, mem_cap_gb: int, timeout_s: int,
            tactile: str = "none") -> dict:
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
            "--tactile", tactile]

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
    ap.add_argument("--tactile", choices=["none", "splat"], default="none")
    args = ap.parse_args()

    ladder = [n for n in DEFAULT_LADDER if n <= args.max_envs]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"scale_sweep_{args.tag}.csv"

    print(f"ramping N={ladder}")
    print(f"cap {args.mem_cap}G per child, {args.steps} timed steps -> {out_csv}\n")
    print(f"{'N':>6}{'env-steps/s':>15}{'us/env-step':>13}{'ms/step':>10}"
          f"{'peak GB':>10}{'con/env':>9}{'ovf':>5}  status")
    print("-" * 74)

    rows = []
    for n in ladder:
        r = run_one(n, args.steps, args.warmup, args.mem_cap, args.timeout,
                    tactile=args.tactile)
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
        if not r.get("ok"):
            print(f"\nceiling reached at N={n}. Stopping.")
            break

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

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
