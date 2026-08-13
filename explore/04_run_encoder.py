"""Run the CPU reference encoder on the model mjlab trains on, and bank an oracle.

This is the first point where touch actually exists on the RL hand: contacts in,
368x3 taxel forces out, produced by the supervisor's own `VirtualTaxelSensor`.

It also writes the oracle fixture that every later implementation -- naive torch,
C++, OpenMP, Triton, CUDA -- must reproduce (plan §7). The fixture stores the
encoder's *inputs* (contact geometry and forces) alongside its outputs, so a GPU
implementation can be checked without re-running MuJoCo at all.

Run:
    PYTHONPATH=third_party/leapXELA_model:explore \
      .venv/bin/python explore/04_run_encoder.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import pathlib

import mujoco as mj
import numpy as np

from leapxela.taxel_layout import N_TAXELS, build_layout
from leapxela.touch_sensor import VirtualTaxelSensor
from taxel_map import BODY_MAP, PINNED_DIR, SUSPECT, remap_layout

OUT = pathlib.Path(__file__).resolve().parent / "out"
SCENE = PINNED_DIR / "scene_mjx_cube.xml"

KERNEL_SIGMA = 0.0035      # defaults from leapxela/visualize_taxel_layout.py
KERNEL_CUTOFF = 0.01

N_STEPS = 600
CLOSE_FRACTION = 0.6
SAMPLE_EVERY = 20          # bank a fixture frame this often


def build_model() -> mj.MjModel:
    spec = mj.MjSpec.from_file(str(SCENE))
    bodies = {b.name: b for b in spec.bodies}
    layout = build_layout()
    for e in layout.entries:
        bodies[BODY_MAP[e.body]].add_site(
            name=e.site_name,
            pos=np.asarray(e.pos, dtype=float),
            quat=np.asarray(e.quat, dtype=float),
            size=[0.0012] * 3,
            group=4,
            rgba=[0.95, 0.15, 0.1, 1.0] if e.body in SUSPECT else [0.1, 0.9, 0.2, 1.0],
        )
    return spec.compile()


def cube_geom_names(model: mj.MjModel) -> list[str]:
    """Named geoms on the manipulated cube (excluding the mocap goal)."""
    names = []
    for gid in range(model.ngeom):
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if "cube" in body.lower() and "goal" not in body.lower() and gname:
            names.append(gname)
    return names


def main() -> None:
    layout = build_layout()
    model = build_model()
    data = mj.MjData(model)

    obj_geoms = cube_geom_names(model)
    print(f"scene: {SCENE.name}")
    print(f"  object geoms treated as 'the cube': {obj_geoms}")

    # The encoder groups taxels by entry.body, which holds tactile-lineage names.
    # Remap them to RL-lineage names or every lookup returns -1 and the encoder
    # silently outputs zeros. See taxel_map.remap_layout.
    sensor = VirtualTaxelSensor(
        model, remap_layout(layout), obj_geoms, KERNEL_SIGMA, KERNEL_CUTOFF
    )
    print(f"  VirtualTaxelSensor constructed OK "
          f"(sigma={KERNEL_SIGMA}, cutoff={KERNEL_CUTOFF})\n")

    lo, hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
    target = lo + CLOSE_FRACTION * (hi - lo)
    mj.mj_forward(model, data)

    print("=" * 74)
    print("THE ENCODER RUNNING — contacts in, 368x3 taxel forces out")
    print("=" * 74)
    print(f"{'step':>6}{'ncon':>6}{'active taxels':>15}{'|F| total (N)':>15}"
          f"{'max taxel (N)':>15}")
    print("-" * 74)

    frames = []
    best = (0, None, None)
    for step in range(N_STEPS):
        data.ctrl[:] = target
        mj.mj_step(model, data)
        taxels = sensor.update(data)                      # (368, 3) float32

        mag = np.linalg.norm(taxels, axis=1)
        active = int((mag > 1e-6).sum())

        if step % SAMPLE_EVERY == 0:
            frames.append(capture(model, data, taxels, obj_geoms))
            if active:
                print(f"{step:>6}{data.ncon:>6}{active:>15}"
                      f"{np.linalg.norm(sensor.last_total_force):>15.4f}"
                      f"{mag.max():>15.4f}")
        if active > best[0]:
            best = (active, taxels.copy(), step)

    print("-" * 74)
    n_active, taxels, at_step = best
    print(f"busiest frame: step {at_step}, {n_active} taxels active\n")

    if taxels is None:
        raise SystemExit("no taxel ever fired — check obj_geoms and kernel_cutoff")

    # ---- where did the signal land? ----------------------------------------
    print("=" * 74)
    print("WHICH PART OF THE HAND FELT IT (busiest frame)")
    print("=" * 74)
    mag = np.linalg.norm(taxels, axis=1)
    for src in BODY_MAP:
        idx = [i for i, e in enumerate(layout.entries) if e.body == src]
        m = mag[idx]
        n = int((m > 1e-6).sum())
        if n:
            flag = "  (fingertip: placement unverified, D4)" if src in SUSPECT else ""
            print(f"  {src:<28} {n:>3}/{len(idx):<3} active   "
                  f"peak {m.max():.4f} N{flag}")

    print(f"\n  channels: shear_x {taxels[:, 0].min():+.3f}..{taxels[:, 0].max():+.3f}   "
          f"shear_y {taxels[:, 1].min():+.3f}..{taxels[:, 1].max():+.3f}   "
          f"normal_z {taxels[:, 2].min():+.3f}..{taxels[:, 2].max():+.3f}")
    print("  (normal_z positive = compression, per the encoder's sign convention)\n")

    # ---- bank the oracle ----------------------------------------------------
    OUT.mkdir(exist_ok=True)
    path = OUT / "oracle_fixture.npz"
    save_fixture(path, frames)
    print(f"oracle fixture -> {path}")
    print(f"  {len(frames)} frames, including ncon=0 cases")
    print("  every later implementation must reproduce these outputs from these inputs")


def capture(model, data, taxels, obj_geoms) -> dict:
    """Everything the encoder consumed this step, plus what it produced."""
    ncon = data.ncon
    force6 = np.zeros(6)
    con_pos, con_frame, con_g1, con_g2, con_force = [], [], [], [], []
    for i in range(ncon):
        c = data.contact[i]
        mj.mj_contactForce(model, data, i, force6)
        con_pos.append(c.pos.copy())
        con_frame.append(c.frame.copy())
        con_g1.append(c.geom1)
        con_g2.append(c.geom2)
        con_force.append(force6.copy())
    return dict(
        time=data.time,
        ncon=ncon,
        contact_pos=np.array(con_pos).reshape(-1, 3),
        contact_frame=np.array(con_frame).reshape(-1, 9),
        contact_geom1=np.array(con_g1, dtype=np.int32),
        contact_geom2=np.array(con_g2, dtype=np.int32),
        contact_force=np.array(con_force).reshape(-1, 6),
        site_xpos=data.site_xpos.copy(),
        site_xmat=data.site_xmat.copy(),
        qpos=data.qpos.copy(),
        taxels=taxels.copy(),
    )


def save_fixture(path: pathlib.Path, frames: list[dict]) -> None:
    flat = {}
    for i, f in enumerate(frames):
        for k, v in f.items():
            flat[f"f{i:03d}_{k}"] = v
    flat["n_frames"] = np.array(len(frames))
    flat["n_taxels"] = np.array(N_TAXELS)
    flat["kernel_sigma"] = np.array(KERNEL_SIGMA)
    flat["kernel_cutoff"] = np.array(KERNEL_CUTOFF)
    np.savez_compressed(path, **flat)


if __name__ == "__main__":
    main()
