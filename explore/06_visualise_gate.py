"""Render the grasp so the taxel gate is visible: who fires, and where contacts land.

Answers a question the numbers alone do not: *why* do whole groups of taxels stay
silent? Rendering makes it obvious that it is geometry, not a bug — the cube
touches faces of the finger links that carry no pads.

Colour code on the taxel sites:
    green  = fired this frame
    grey   = present but silent
Contact points are drawn by MuJoCo's own contact visualisation (red discs).

Headless: writes PNGs, no interactive viewer needed.

Run:
    PYTHONPATH=third_party/leapXELA_model:explore \\
      .venv/bin/python explore/06_visualise_gate.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import pathlib

import mujoco as mj
import numpy as np
from PIL import Image

from leapxela.taxel_layout import build_layout
from leapxela.touch_sensor import VirtualTaxelSensor
from taxel_map import BODY_MAP, PINNED_DIR, remap_layout

OUT = pathlib.Path(__file__).resolve().parent / "out" / "gate"
SCENE = PINNED_DIR / "scene_mjx_cube.xml"

KERNEL_SIGMA, KERNEL_CUTOFF = 0.0035, 0.01
SIM_DT = 0.01
CLOSE_FRACTION = 0.6
SETTLE = 250

LIVE = [0.1, 0.95, 0.2, 1.0]      # fired
DARK = [0.45, 0.45, 0.5, 0.85]    # silent


def build():
    spec = mj.MjSpec.from_file(str(SCENE))
    bodies = {b.name: b for b in spec.bodies}
    for e in build_layout().entries:
        bodies[BODY_MAP[e.body]].add_site(
            name=e.site_name,
            pos=np.asarray(e.pos, float),
            quat=np.asarray(e.quat, float),
            size=[0.0022] * 3,          # exaggerated so the pads read at a glance
            group=4,
            rgba=DARK,
        )
    m = spec.compile()
    m.opt.timestep = SIM_DT
    return m


def main() -> None:
    layout = build_layout()
    model = build()
    data = mj.MjData(model)

    obj = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g)
        for g in range(model.ngeom)
        if "cube" in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").lower()
        and "goal" not in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").lower()
        and mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g)
    ]
    sensor = VirtualTaxelSensor(model, remap_layout(layout), obj, KERNEL_SIGMA, KERNEL_CUTOFF)
    site_id = {e.site_name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, e.site_name)
               for e in layout.entries}

    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    target = lo + CLOSE_FRACTION * (hi - lo)
    mj.mj_forward(model, data)
    for _ in range(SETTLE):
        data.ctrl[:] = target
        mj.mj_step(model, data)

    taxels = sensor.update(data)
    live = np.linalg.norm(taxels, axis=1) > 1e-6
    print(f"contacts: {data.ncon}   taxels firing: {live.sum()}/368")

    # colour the sites by whether they fired
    for i, e in enumerate(layout.entries):
        model.site_rgba[site_id[e.site_name]] = LIVE if live[i] else DARK

    # which bodies own the contacts, and which own the firing taxels
    import collections
    cb = collections.Counter()
    for i in range(data.ncon):
        c = data.contact[i]
        for g in (c.geom1, c.geom2):
            n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
            if "cube" not in n.lower():
                cb[n] += 1
    print("contacts by hand body:", dict(cb))
    fb = collections.Counter(BODY_MAP[layout.entries[i].body] for i in np.nonzero(live)[0])
    print("firing taxels by body:", dict(fb))
    print("\n-> bodies with contacts but NO firing taxels are the gate at work:")
    for b in cb:
        if b and b not in fb:
            print(f"     {b}: {cb[b]} contacts, 0 taxels fired")

    OUT.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 1200
    model.vis.headlight.ambient[:] = [0.5] * 3
    model.vis.headlight.diffuse[:] = [0.75] * 3
    model.vis.scale.contactwidth = 0.12
    model.vis.scale.contactheight = 0.05

    opt = mj.MjvOption()
    opt.sitegroup[4] = 1
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1     # show contacts

    base = mj.MjvCamera()
    mj.mjv_defaultFreeCamera(model, base)
    views = {
        "01_overview": dict(az=115, el=-30, zoom=0.60),
        "02_palm_down": dict(az=90, el=-88, zoom=0.52),
        "03_fingertips": dict(az=138, el=-38, zoom=0.40),
        "04_thumb_live": dict(az=52, el=-30, zoom=0.34),
    }
    with mj.Renderer(model, height=1200, width=1600) as r:
        for name, v in views.items():
            c = mj.MjvCamera()
            c.type = mj.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = base.lookat
            c.distance = base.distance * v["zoom"]
            c.azimuth, c.elevation = v["az"], v["el"]
            r.update_scene(data, camera=c, scene_option=opt)
            p = OUT / f"{name}.png"
            Image.fromarray(r.render()).save(p)
            print("wrote", p)


if __name__ == "__main__":
    main()
