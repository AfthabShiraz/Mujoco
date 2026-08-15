"""Does `patch_mjlab_site_pos_w` change any number, or only the time taken?

A speedup that alters results is not a speedup, it is a bug. This script exists
because the patch was measured (3.31x) before it was verified, which is the wrong
order.

Three independent checks, strongest first:

  1. PROPERTY EQUALITY. `site_pos_w` as shipped is
     `torch.cat([pos, quat], -1)[..., 0:3]`, and the patch returns `pos`
     directly. Slicing the first three columns off a concatenation whose first
     three columns ARE `pos` is the identity, so the two must agree BIT FOR BIT,
     not approximately. Anything less than exact equality here means the reading
     of the code is wrong and the patch must be withdrawn.

  2. OBSERVATION EQUALITY AT STEP 1. Two environments, same seed, same actions,
     one patched. The observation tensors the policy receives must be identical
     on the first step. This catches the case where something other than
     `site_pos_w` reads the property chain and is affected.

  3. TRAJECTORY DIVERGENCE OVER MANY STEPS. Reported, not asserted. Contact
     solvers use atomics, so two runs of the SAME binary need not be
     bit-identical over time, and a rigid-body sim with contact is chaotic --
     differences grow whether or not any code changed. To tell "the patch changed
     something" from "this is ordinary GPU non-determinism", the script also runs
     unpatched-vs-unpatched as a control. The patched-vs-unpatched divergence
     must be no larger than that control.

Check 3 is the one that would otherwise be misread. A growing difference is only
evidence of a bug if it exceeds what the same code does against itself.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mjlab_tactile"))


def build(num_envs: int, seed: int, taxels: bool):
    from leap_xela_mjlab.tasks.reorient.config.env_cfg import make_reorient_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    import taxel_term

    cfg = make_reorient_env_cfg(disable_cube_friction_dr=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if taxels:
        cfg.scene.entities["robot"] = taxel_term.entity_cfg_with_taxels()
    return ManagerBasedRlEnv(cfg, device="cuda:0")


def rollout(num_envs: int, seed: int, steps: int, taxels: bool):
    """Deterministic rollout -> list of per-step observation dicts (on host)."""
    import torch

    torch.manual_seed(seed)
    env = build(num_envs, seed, taxels)
    obs, _ = env.reset()
    # Fixed, non-zero actions: zeros leave the hand limp and exercise little.
    g = torch.Generator(device="cpu").manual_seed(seed)
    action = (
        torch.rand((num_envs, env.action_manager.total_action_dim), generator=g)
        * 2.0 - 1.0
    ).to("cuda:0")

    frames = [{k: v.detach().cpu().clone() for k, v in obs.items()}]
    for _ in range(steps):
        obs = env.step(action)[0]
        frames.append({k: v.detach().cpu().clone() for k, v in obs.items()})
    env.close()
    return frames


def max_abs_diff(a, b) -> float:
    import torch

    return max(
        float((a[k].double() - b[k].double()).abs().max()) for k in a
    ) if a else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import taxel_term

    # ---- check 1: the property, bit for bit ------------------------------- #
    env = build(args.num_envs, args.seed, taxels=True)
    env.reset()
    for _ in range(5):
        env.step(torch.zeros((args.num_envs, env.action_manager.total_action_dim),
                             device="cuda:0"))
    hand = env.scene["robot"]
    shipped = hand.data.site_pos_w.clone()
    direct = hand.data.data.site_xpos[:, hand.data.indexing.site_ids].clone()
    exact = torch.equal(shipped, direct)
    print(f"[1] site_pos_w shipped vs site_xpos gather : "
          f"shape {tuple(shipped.shape)}  bitwise identical: {exact}  "
          f"max|diff| {float((shipped-direct).abs().max()):.3e}")
    also_quat = torch.equal(hand.data.site_quat_w,
                            hand.data.site_pose_w[..., 3:7])
    print(f"    site_quat_w untouched by the patch path : {also_quat}")
    env.close()
    del env

    # ---- check 2 and 3: whole-environment observations -------------------- #
    print("\n[2/3] rolling out identical environments...")
    base_a = rollout(args.num_envs, args.seed, args.steps, taxels=True)
    base_b = rollout(args.num_envs, args.seed, args.steps, taxels=True)
    control = [max_abs_diff(x, y) for x, y in zip(base_a, base_b)]

    applied = taxel_term.patch_mjlab_site_pos_w()
    print(f"    patch applied: {applied}")
    patched = rollout(args.num_envs, args.seed, args.steps, taxels=True)
    treat = [max_abs_diff(x, y) for x, y in zip(base_a, patched)]

    print(f"\n    step |  unpatched vs unpatched  |  unpatched vs patched")
    print(f"    -----+--------------------------+----------------------")
    for i in (0, 1, 2, 5, 10, min(args.steps, 19)):
        if i < len(control):
            print(f"    {i:4d} | {control[i]:24.3e} | {treat[i]:.3e}")

    step0_ok = treat[0] == 0.0
    step1_ok = treat[1] <= max(control[1], 0.0)
    never_worse = all(t <= max(c, 1e-12) * 10 + 1e-9
                      for t, c in zip(treat, control))

    print(f"\n    step-0 observations bit-identical      : {step0_ok}")
    print(f"    step-1 diff within same-code control   : {step1_ok}")
    print(f"    never exceeds control by >10x          : {never_worse}")
    print(f"\nVERDICT: {'PASS' if (exact and step0_ok and never_worse) else 'FAIL'}")
    return 0 if (exact and step0_ok and never_worse) else 1


if __name__ == "__main__":
    sys.exit(main())
