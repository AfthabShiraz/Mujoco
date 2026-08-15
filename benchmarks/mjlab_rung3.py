"""Cost of tactile observations inside the supervisor's real training environment.

Everything in `scale_sweep.py` measures a bare `mjw.step` loop: no policy, no
rewards, no episode resets, no domain randomisation, and one control update per
physics step. That is the right shape for comparing four tactile representations
against each other, because it holds everything else still. It is the wrong shape
for answering "what does this cost Hamid", because his wall-clock also contains
the observation managers, the reward and termination managers, the command
resampler, the event system, five physics steps per control step, and -- in real
training -- a policy and an optimiser.

This script closes that gap. It builds `Mjlab-LeapXELA-Cube-Reorient` through his
own `make_reorient_env_cfg`, optionally adds the 368-taxel splat observation, and
times `env.step`.

WHAT IT DOES AND DOES NOT MEASURE.
  * It DOES include every part of the mjlab environment: managers, resets,
    randomisation, decimation.
  * It DOES NOT include the policy forward pass or the PPO update. Actions are a
    fixed zero tensor. So the tactile share reported here is an UPPER BOUND on
    the share it would occupy in real training -- adding the network and
    optimiser makes the denominator larger and the tactile fraction smaller.
  * The contact regime is that of an untrained policy, which mostly drops the
    cube. A trained policy grips deliberately and sustains far more contact, and
    the encoder's cost rises with live contact count. So this UNDERSTATES the
    late-training cost. Report both directions; do not quote one number as "the"
    cost.

UNITS. `env.step` advances `decimation` physics steps (5 in his config). This
script reports env-steps/s in mjlab's sense and, separately, the equivalent
physics steps/s, because the sweep CSVs are in physics steps and the two get
confused otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mjlab_tactile"))


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--tactile", choices=["none", "splat"], default="none")
    ap.add_argument("--encoder", choices=["torch", "compile", "triton"],
                    default="triton")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patch-site-pos", choices=["on", "off"], default="off",
                    help="apply the mjlab `site_pos_w` fix (taxel_term."
                         "patch_mjlab_site_pos_w). Default off so the number "
                         "reflects mjlab as shipped; 'on' shows what the same "
                         "environment costs once that upstream waste is removed")
    args = ap.parse_args()

    import torch
    from leap_xela_mjlab.tasks.reorient.config.env_cfg import make_reorient_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.observation_manager import ObservationTermCfg

    patched = False
    if args.patch_site_pos == "on":
        import taxel_term as _tt
        patched = _tt.patch_mjlab_site_pos_w()

    cfg = make_reorient_env_cfg(disable_cube_friction_dr=True)
    cfg.scene.num_envs = args.num_envs

    n_taxels = 0
    if args.tactile == "splat":
        import taxel_term

        # Swap the hand for one carrying taxel sites, then add the term. Order
        # matters only in that both must happen before the env is constructed.
        cfg.scene.entities["robot"] = taxel_term.entity_cfg_with_taxels()

        def _taxels(env):
            return taxel_term.taxel_forces(env, encoder=args.encoder)

        # Actor only. The critic already receives privileged cube state, so
        # adding touch there would measure cost without testing the thing touch
        # is for.
        cfg.observations["actor"].terms["taxel_forces"] = ObservationTermCfg(
            func=_taxels
        )
        n_taxels = 368

    t0 = time.perf_counter()
    env = ManagerBasedRlEnv(cfg, device=args.device)
    setup_s = time.perf_counter() - t0

    obs, _ = env.reset()
    shapes = {k: tuple(v.shape) for k, v in obs.items()}

    action = torch.zeros(
        (args.num_envs, env.action_manager.total_action_dim), device=args.device
    )

    for _ in range(args.warmup):
        env.step(action)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(action)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    d = env.sim.wp_data
    nacon = int(d.nacon.numpy()[0])
    decim = int(cfg.decimation)

    result = dict(
        num_envs=args.num_envs,
        tactile=args.tactile,
        site_pos_patched=patched,
        encoder=args.encoder if args.tactile == "splat" else "",
        n_taxels=n_taxels,
        steps=args.steps,
        decimation=decim,
        env_steps_per_sec=args.num_envs * args.steps / elapsed,
        physics_steps_per_sec=args.num_envs * args.steps * decim / elapsed,
        ms_per_env_step=1000.0 * elapsed / args.steps,
        ms_per_physics_step=1000.0 * elapsed / (args.steps * decim),
        obs_shapes=shapes,
        contacts=nacon,
        contacts_per_env=nacon / args.num_envs,
        naconmax_total=int(d.naconmax),
        overflow_worlds=int((d.overflow.numpy() != 0).sum()),
        setup_s=setup_s,
        wall_s=elapsed,
        peak_rss_gb=peak_rss_gb(),
        torch_peak_gb=torch.cuda.max_memory_allocated() / 1024**3,
        ok=True,
    )
    print(json.dumps(result))
    print(
        f"[rung3] N={args.num_envs} tactile={args.tactile} "
        f"patch={args.patch_site_pos} "
        f"{result['env_steps_per_sec']:,.0f} env-steps/s "
        f"({result['physics_steps_per_sec']:,.0f} physics-steps/s)  "
        f"{result['ms_per_env_step']:.2f} ms/env-step  "
        f"peak {result['peak_rss_gb']:.2f} GB  "
        f"contacts {result['contacts_per_env']:.1f}/env  "
        f"obs {shapes}",
        file=sys.stderr,
    )
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
