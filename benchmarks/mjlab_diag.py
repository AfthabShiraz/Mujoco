"""Why does the splat encoder cost more inside mjlab than in the harness?

`mjlab_rung3.py` prices tactile as a difference of two whole-environment step
times. That is the number that matters, but it cannot say WHERE the time goes,
and at N=2048 it implies ~35 ms per encode against ~1.5 ms for the same encoder
on the same GPU in `benchmarks/harness.py`. A 20x discrepancy is a bug or a
misunderstanding, not a measurement.

This script takes the environment apart and times the pieces separately:

  step_only     env.step() with the taxel term absent      -- mjlab's own cost
  encode_only   enc.encode() called directly, in isolation -- the encoder's cost
  term_only     the observation term, including its stream wait
  obs_compute   the whole observation manager
  full          env.step() with the term present

If `encode_only` is small and `full - step_only` is large, the cost is in the
integration (stream ordering, extra invocations, observation plumbing) and is
fixable. If `encode_only` is itself large, the encoder genuinely behaves
differently here and the harness number does not transfer.

It also counts how many times the term is actually invoked per `env.step`, which
is the first thing to rule out: mjlab computes observation groups on its own
schedule, and a term called twice per step costs twice as much for reasons that
have nothing to do with the kernel.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mjlab_tactile"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--encoder", default="triton")
    args = ap.parse_args()

    import torch
    import warp as wp
    from leap_xela_mjlab.tasks.reorient.config.env_cfg import make_reorient_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.observation_manager import ObservationTermCfg

    import taxel_term

    calls = {"n": 0}

    cfg = make_reorient_env_cfg(disable_cube_friction_dr=True)
    cfg.scene.num_envs = args.num_envs
    cfg.scene.entities["robot"] = taxel_term.entity_cfg_with_taxels()

    def _taxels(env):
        calls["n"] += 1
        return taxel_term.taxel_forces(env, encoder=args.encoder)

    cfg.observations["actor"].terms["taxel_forces"] = ObservationTermCfg(func=_taxels)

    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    action = torch.zeros(
        (args.num_envs, env.action_manager.total_action_dim), device="cuda:0"
    )

    def sync():
        wp.synchronize()
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        env.step(action)
    sync()

    # -- how many times is the term invoked per env.step? --------------------
    calls["n"] = 0
    for _ in range(10):
        env.step(action)
    sync()
    per_step = calls["n"] / 10.0

    def timed(fn, n):
        sync()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return 1000.0 * (time.perf_counter() - t) / n

    full_ms = timed(lambda: env.step(action), args.steps)
    obs_ms = timed(lambda: env.observation_manager.compute(), args.steps)
    term_ms = timed(lambda: taxel_term.taxel_forces(env, encoder=args.encoder),
                    args.steps)

    enc = taxel_term._CACHE.get(env, args.encoder)
    enc_ms = timed(enc.encode, args.steps)

    # The term without its stream wait: isolates the ordering cost.
    def _no_wait():
        return enc.encode().reshape(args.num_envs, -1)

    nowait_ms = timed(_no_wait, args.steps)

    print(f"\n=== N={args.num_envs}  encoder={args.encoder} ===")
    print(f"  term invocations per env.step : {per_step:.1f}")
    print(f"  full  env.step()              : {full_ms:8.2f} ms")
    print(f"  obs_manager.compute()         : {obs_ms:8.2f} ms")
    print(f"  taxel term (with stream wait) : {term_ms:8.2f} ms")
    print(f"  encode() + reshape, no wait   : {nowait_ms:8.2f} ms")
    print(f"  encode() alone                : {enc_ms:8.2f} ms")
    print(f"  -> stream-wait overhead       : {term_ms - nowait_ms:8.2f} ms")
    print(f"  -> obs plumbing over term     : {obs_ms - term_ms:8.2f} ms")
    print(f"  -> step minus obs             : {full_ms - obs_ms:8.2f} ms")
    d = env.sim.wp_data
    print(f"  naconmax {int(d.naconmax)}  live contacts {int(d.nacon.numpy()[0])}"
          f"  C per env {enc.C}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
