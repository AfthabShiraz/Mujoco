"""See MuJoCo contacts appear and disappear as the hand closes on the cube.

Learning script, not part of the benchmark suite. CPU MuJoCo, one environment,
headless. Nothing here touches the GPU.

The point: MuJoCo tells you "there is a force of F at this point in space,
between geom A and geom B". It does NOT tell you "taxel 147 felt 0.3 N".
Watching data.ncon change is how that gap becomes concrete.

Run:
    systemd-run --user --scope -q -p MemoryMax=8G .venv/bin/python explore/01_see_contacts.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"  # headless offscreen rendering; must precede import

import pathlib

import mujoco as mj
import numpy as np

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "leapXELA_model"
SCENE = MODEL_DIR / "scene_mjx_cube.xml"
OUT = pathlib.Path(__file__).resolve().parent / "out"

N_STEPS = 600
CLOSE_FRACTION = 0.6  # how far toward each joint's upper limit to drive the fingers


def describe_model(model: mj.MjModel) -> None:
    print("=" * 72)
    print("THE MODEL — the parts that never change")
    print("=" * 72)
    print(f"  bodies   {model.nbody:5d}   rigid parts")
    print(f"  geoms    {model.ngeom:5d}   collision/visual shapes")
    print(f"  sites    {model.nsite:5d}   massless labelled frames (taxels would live here)")
    print(f"  joints   {model.njnt:5d}")
    print(f"  dofs     {model.nv:5d}   degrees of freedom")
    print(f"  actuators{model.nu:5d}")
    print(f"  sensors  {model.nsensor:5d}")
    print()

    site_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
    taxel_sites = [n for n in site_names if n and "taxel" in n.lower()]
    print(f"  sites with 'taxel' in the name: {len(taxel_sites)}")
    print("  -> this is the missing-taxel-sites problem (PROJECT_LOG §1.5b):")
    print("     the model the GPU stack loads has no taxel sites at all.")
    print()


def cube_geom_ids(model: mj.MjModel) -> set[int]:
    """Geoms belonging to any body whose name mentions the cube."""
    ids = set()
    for gid in range(model.ngeom):
        body = model.geom_bodyid[gid]
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body) or ""
        if "cube" in name.lower() and "goal" not in name.lower():
            ids.add(gid)
    return ids


def main() -> None:
    if not SCENE.exists():
        raise SystemExit(f"Scene not found: {SCENE}")

    model = mj.MjModel.from_xml_path(str(SCENE))
    data = mj.MjData(model)
    describe_model(model)

    cube_geoms = cube_geom_ids(model)
    print(f"  cube geoms: {sorted(cube_geoms)}\n")

    # Drive every actuator toward CLOSE_FRACTION of its control range, so the
    # fingers curl onto the cube and contacts actually form.
    lo, hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
    target = lo + CLOSE_FRACTION * (hi - lo)

    mj.mj_forward(model, data)

    print("=" * 72)
    print("THE STATE — stepping, watching contacts form")
    print("=" * 72)
    print(f"{'step':>6} {'time':>7} {'ncon':>5}  {'hand-cube':>9}  note")
    print("-" * 72)

    peak = 0
    for step in range(N_STEPS):
        data.ctrl[:] = target
        mj.mj_step(model, data)

        hand_cube = sum(
            1
            for i in range(data.ncon)
            for c in [data.contact[i]]
            if (c.geom1 in cube_geoms) != (c.geom2 in cube_geoms)  # exactly one side is cube
        )
        peak = max(peak, data.ncon)

        if step % 60 == 0 or (hand_cube and step % 20 == 0):
            note = "cube touched" if hand_cube else ""
            print(f"{step:6d} {data.time:7.3f} {data.ncon:5d}  {hand_cube:9d}  {note}")

    print("-" * 72)
    print(f"peak ncon over the run: {peak}")
    print(f"the GPU config caps this at nconmax=48 per env -> {'FITS' if peak <= 48 else 'OVERFLOWS'}")
    print()

    # ---- inspect one contact in full ---------------------------------------
    print("=" * 72)
    print("ONE CONTACT, IN FULL — this is the encoder's raw input")
    print("=" * 72)
    force6 = np.zeros(6)
    shown = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in cube_geoms) == (c.geom2 in cube_geoms):
            continue
        mj.mj_contactForce(model, data, i, force6)
        g1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom2)
        b1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
        b2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
        frame = c.frame.reshape(3, 3)
        print(f"  contact {i}")
        print(f"    geoms      {g1}  <->  {g2}")
        print(f"    bodies     {b1}  <->  {b2}")
        print(f"    position   {np.array2string(c.pos, precision=4)}   (world frame)")
        print(f"    penetration{c.dist: .6f} m  (negative = overlapping)")
        print(f"    force      normal={force6[0]: .4f} N  tan1={force6[1]: .4f}  tan2={force6[2]: .4f}")
        print(f"    world force{np.array2string(frame.T @ force6[:3], precision=4)}")
        print("    ^ the encoder turns exactly this into per-taxel forces")
        shown += 1
        if shown >= 3:
            break
    if not shown:
        print("  (no hand-cube contacts at the final step — try raising CLOSE_FRACTION)")
    print()

    # ---- render proof ------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    # The offscreen framebuffer defaults to 640x480 (set in the model XML's
    # <visual><global offwidth/offheight>). Raise it before making a Renderer.
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 600
    with mj.Renderer(model, height=600, width=800) as r:
        r.update_scene(data, camera="side")
        img = r.render()
    from PIL import Image

    path = OUT / "final_pose.png"
    Image.fromarray(img).save(path)
    print(f"rendered final pose -> {path}  (open it in VS Code)")


if __name__ == "__main__":
    main()
