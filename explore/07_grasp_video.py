"""Render the whole grasp as a video, with taxels lighting up as they fire.

The still renders are hard to read: the hand is dark, the pads are small, and a
single frame cannot show a taxel switching on. A video fixes all three — you can
watch the fingers close, contacts appear, and see which pads respond and which
stay dark while being pressed.

Deliberate visual choices, all to make the pads legible:
  * taxels drawn far larger than life (3.5 mm vs the real 1.2 mm)
  * the hand shell is lightened so dark pads read against it
  * firing taxels go bright green and are scaled up further
  * contact forces drawn as arrows

Headless: writes an MP4, no interactive viewer needed.

Run:
    PYTHONPATH=third_party/leapXELA_model:explore \\
      .venv/bin/python explore/07_grasp_video.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import pathlib

import imageio.v2 as imageio
import mujoco as mj
import numpy as np

from leapxela.taxel_layout import build_layout
from leapxela.touch_sensor import VirtualTaxelSensor
from taxel_map import BODY_MAP, PINNED_DIR, remap_layout

OUT = pathlib.Path(__file__).resolve().parent / "out" / "gate"
SCENE = PINNED_DIR / "scene_mjx_cube.xml"

KERNEL_SIGMA, KERNEL_CUTOFF = 0.0035, 0.01
SIM_DT = 0.01
N_STEPS = 400
FPS = 25
EVERY = 2                      # render every Nth step

SITE_R = 0.0035                # exaggerated; real pads are 1.2 mm
LIVE = np.array([0.15, 1.0, 0.25, 1.0])
DARK = np.array([0.30, 0.33, 0.40, 1.0])


def build() -> mj.MjModel:
    spec = mj.MjSpec.from_file(str(SCENE))
    bodies = {b.name: b for b in spec.bodies}
    for e in build_layout().entries:
        bodies[BODY_MAP[e.body]].add_site(
            name=e.site_name,
            pos=np.asarray(e.pos, float),
            quat=np.asarray(e.quat, float),
            size=[SITE_R] * 3,
            group=4,
            rgba=DARK,
        )
    m = spec.compile()
    m.opt.timestep = SIM_DT
    # Lighten the hand so the pads stand out against it.
    for g in range(m.ngeom):
        body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if "cube" not in body.lower():
            m.geom_matid[g] = -1                       # drop the dark material
            m.geom_rgba[g] = [0.78, 0.80, 0.84, 1.0]   # so pads read against it
        else:
            # The cube sits ON the palm, hiding the 120 taxels underneath it.
            # Make it glassy so you can watch those pads light up through it.
            m.geom_matid[g] = -1
            m.geom_rgba[g] = [0.30, 0.75, 0.95, 0.28]
    return m


def main() -> None:
    layout = build_layout()
    model = build()
    data = mj.MjData(model)

    obj = [
        n for g in range(model.ngeom)
        if (n := mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g))
        and "cube" in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").lower()
        and "goal" not in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").lower()
    ]
    sensor = VirtualTaxelSensor(model, remap_layout(layout), obj, KERNEL_SIGMA, KERNEL_CUTOFF)
    sids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, e.site_name)
                     for e in layout.entries])

    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    target = lo + 0.6 * (hi - lo)

    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
    model.vis.headlight.ambient[:] = [0.55] * 3
    model.vis.headlight.diffuse[:] = [0.8] * 3
    model.vis.scale.contactwidth = 0.10
    model.vis.scale.contactheight = 0.03
    model.vis.scale.forcewidth = 0.04

    opt = mj.MjvOption()
    opt.sitegroup[4] = 1
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

    mj.mj_forward(model, data)
    base = mj.MjvCamera()
    mj.mjv_defaultFreeCamera(model, base)

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = base.lookat
    # Look straight down into the palm: that is where 120 of the 368 taxels are,
    # and where most of the firing happens. The orbiting wide shot hid both.
    cam.distance = base.distance * 0.42
    cam.elevation = -72
    cam.azimuth = 90

    OUT.mkdir(parents=True, exist_ok=True)
    frames, peak = [], 0
    with mj.Renderer(model, height=960, width=1280) as r:
        for step in range(N_STEPS):
            data.ctrl[:] = target
            mj.mj_step(model, data)
            if step % EVERY:
                continue

            t = sensor.update(data)
            mag = np.linalg.norm(t, axis=1)
            live = mag > 1e-6
            peak = max(peak, int(live.sum()))

            model.site_rgba[sids] = np.where(live[:, None], LIVE, DARK)
            # grow a firing taxel so it reads even at a distance
            model.site_size[sids] = SITE_R * np.where(live, 2.0, 1.0)[:, None]

            r.update_scene(data, camera=cam, scene_option=opt)
            frames.append(r.render())

    path = OUT / "grasp_palm_xray.mp4"
    imageio.mimwrite(path, frames, fps=FPS, quality=8, macro_block_size=1)
    print(f"wrote {path}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")
    print(f"peak taxels firing in any frame: {peak}/368")


if __name__ == "__main__":
    main()
