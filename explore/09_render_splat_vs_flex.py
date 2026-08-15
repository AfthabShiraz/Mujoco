"""Render the splat/flex disagreement, so it can be seen rather than argued about.

`08_splat_vs_flex.py` measured it: driven by the same grasp, the flex skin
reports contact on 15 of 18 patches while the splat encoder reports it on 3 --
the palm only. Every fingertip is touching the cube and none of its 120 taxels
fire, because the RL-lineage tip is ~10 mm longer than the tip the taxels were
calibrated against (`taxel_map.SUSPECT`, D4), so contacts land 9-10.8 mm out of
the taxel plane against a 10 mm `kernel_cutoff`.

That is a geometric claim, and geometry is worth looking at.

Two images, answering two different questions:

  A_consequence.png -- WHAT GOES WRONG. The two hands at the same grasp, side by
      side. Left: the rigid hand the splat runs on, taxel pads coloured green if
      they fired and grey if silent, MuJoCo's red discs marking real contacts.
      Right: the flex hand, its skin vertices coloured by tared Kelvin-Voigt
      force. The fingertips carry red contact discs and grey pads on the left,
      and lit skin on the right.

  B_cause.png -- WHY. A close-up of one fingertip on the rigid hand. The red
      contact disc sits visibly off the grey pad cluster; the gap it stands off
      by is the ~10 mm that the gate rejects.

Colour code follows `06_visualise_gate.py` exactly, so the two sets of renders
can be read together:
    green = fired this frame,  grey = present but silent,  red disc = contact.

Headless via EGL. Run:
    PYTHONPATH=third_party/leapXELA_model:explore \\
      .venv/bin/python explore/09_render_splat_vs_flex.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"   # must precede `import mujoco`

import pathlib
import sys

import mujoco as mj
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "explore" / "out" / "gate"
for _p in (ROOT / "benchmarks", ROOT / "explore", ROOT / "explore" / "vendor",
           ROOT / "third_party" / "leapXELA_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FLEX_SCENE = ROOT / "third_party" / "leapXELA_model" / "scene_flex_sensor_Box.xml"

LIVE = [0.1, 0.95, 0.2, 1.0]      # fired            (matches 06)
DARK = [0.45, 0.45, 0.5, 0.85]    # silent           (matches 06)
HOT = [0.95, 0.35, 0.05, 1.0]     # flex skin, loaded
COOL = [0.45, 0.45, 0.5, 0.85]    # flex skin, unloaded

PREGRIP_S = 3.0
GRASP_S = 3.0
ACTIVE_N = 1e-3


def _load(path: pathlib.Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _shoot(model, data, path, az, el, dist, w=1600, h=1200, lookat=None,
           flex=False):
    model.vis.global_.offwidth, model.vis.global_.offheight = w, h
    model.vis.headlight.ambient[:] = [0.5] * 3
    model.vis.headlight.diffuse[:] = [0.75] * 3
    model.vis.scale.contactwidth = 0.12
    model.vis.scale.contactheight = 0.05

    opt = mj.MjvOption()
    opt.sitegroup[4] = 1
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    if flex:
        # Flexes are not drawn by default in the same way rigid geoms are; the
        # skin/face flags are what make the deformable patches visible at all.
        for f in ("mjVIS_FLEXSKIN", "mjVIS_FLEXFACE", "mjVIS_FLEXEDGE",
                  "mjVIS_FLEXVERT"):
            if hasattr(mj.mjtVisFlag, f):
                opt.flags[getattr(mj.mjtVisFlag, f)] = 1

    # ABSOLUTE camera, identical for both models. `mjv_defaultFreeCamera` sizes
    # itself from the model extent, and the flex hand (388 bodies) has a wildly
    # different extent from the rigid one (20) -- using it framed the flex render
    # on empty floor. Since both hands' palms were aligned to the same world
    # pose, a fixed lookat and distance frames them identically.
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    if lookat is None:
        # 06_visualise_gate.py's framing, which is known to show the fingertip
        # pads legibly: the model's own default free camera, pulled in by a
        # factor. Absolute distances were tried and are worse -- the cube fills
        # the frame and occludes the very pads the image exists to show.
        base = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(model, base)
        cam.lookat[:] = base.lookat
        cam.distance = base.distance * dist
    else:
        cam.lookat[:] = lookat
        cam.distance = dist
    cam.azimuth, cam.elevation = az, el

    with mj.Renderer(model, height=h, width=w) as r:
        r.update_scene(data, camera=cam, scene_option=opt)
        img = r.render()
    Image.fromarray(img).save(path)
    return img


def main() -> int:
    import harness as H
    import flex_util
    gm = _load(ROOT / "explore" / "05_grasp_motions.py", "grasp_motions")
    from leapxela.taxel_layout import build_layout
    from leapxela.touch_sensor import VirtualTaxelSensor
    from taxel_map import BODY_MAP, remap_layout

    OUT.mkdir(parents=True, exist_ok=True)
    ref = mj.MjModel.from_xml_path(str(H.DEFAULT_SCENE))

    # ---- flex hand, with the same cube, same alignment as 08 ---------------- #
    fspec = mj.MjSpec.from_file(str(FLEX_SCENE))
    fb = {b.name: b for b in fspec.bodies}
    palm_ref = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_BODY, "palm")
    fb["palm"].pos = ref.body_pos[palm_ref].copy()
    fb["palm"].quat = ref.body_quat[palm_ref].copy()
    svf = _load(ROOT / "explore" / "08_splat_vs_flex.py", "svf")
    svf._add_cube_only(fspec, ref)
    fm = fspec.compile()
    dt = float(fm.opt.timestep)          # 0.002 -- the flex model's own
    fd = mj.MjData(fm)

    # ---- rigid hand with taxel SITES drawn large enough to read ------------- #
    rspec = mj.MjSpec.from_file(str(H.DEFAULT_SCENE))
    rb = {b.name: b for b in rspec.bodies}
    layout_raw = build_layout()
    for e in layout_raw.entries:
        rb[BODY_MAP[e.body]].add_site(
            name=e.site_name, pos=np.asarray(e.pos, float),
            quat=np.asarray(e.quat, float),
            size=[0.0022] * 3,           # exaggerated, as in 06
            group=4, rgba=DARK,
        )
    rm = rspec.compile()
    rm.opt.timestep = dt
    rd = mj.MjData(rm)
    layout = remap_layout(layout_raw)
    splat = VirtualTaxelSensor(rm, layout, H.cube_geom_names(rm),
                               H.KERNEL_SIGMA, H.KERNEL_CUTOFF)
    site_id = {e.site_name: mj.mj_name2id(rm, mj.mjtObj.mjOBJ_SITE, e.site_name)
               for e in layout.entries}

    # ---- drive both to the same grasp, same absolute joint angles ----------- #
    profile = gm.GraspProfile(pattern="hold")
    ridx = np.array([mj.mj_name2id(rm, mj.mjtObj.mjOBJ_ACTUATOR,
                                   mj.mj_id2name(fm, mj.mjtObj.mjOBJ_ACTUATOR, i))
                     for i in range(fm.nu)])
    r_lo, r_hi = rm.actuator_ctrlrange[:, 0], rm.actuator_ctrlrange[:, 1]

    def drive(t):
        tf = gm.grasp_target(fm, t, profile)
        fd.ctrl[:] = tf
        rd.ctrl[ridx] = np.clip(tf, r_lo[ridx], r_hi[ridx])

    est = flex_util.AllFlexForceEstimator(fm)
    mj.mj_forward(fm, fd)
    mj.mj_forward(rm, rd)
    drive(0.0)
    for _ in range(int(round(PREGRIP_S / dt))):
        mj.mj_step(fm, fd)
        mj.mj_step(rm, rd)
        est.update(fm, fd)
    base_force = {k: v.copy() for k, v in est.update(fm, fd).items()}
    for k in range(int(round(GRASP_S / dt))):
        drive(k * dt)
        mj.mj_step(fm, fd)
        mj.mj_step(rm, rd)
        raw = est.update(fm, fd)

    # ---- colour the rigid hand's taxels by whether they fired --------------- #
    out = splat.update(rd)
    live = np.linalg.norm(out, axis=1) > ACTIVE_N
    for i, e in enumerate(layout.entries):
        rm.site_rgba[site_id[e.site_name]] = LIVE if live[i] else DARK

    # ---- colour the flex hand's skin by tared force ------------------------- #
    tared = {k: raw[k] - base_force[k] for k in raw}
    lit_v = 0
    for fid in range(fm.nflex):
        name = mj.mj_id2name(fm, mj.mjtObj.mjOBJ_FLEX, fid)
        mag = np.linalg.norm(tared[name], axis=1)
        # Flex is shaded per-flex, not per-vertex: MuJoCo colours a flex as a
        # whole, so "lit" means this patch carries load somewhere on it. That is
        # the same granularity the measurement reported (per patch), so the
        # picture and the table agree by construction.
        hot = float(mag.max()) > ACTIVE_N
        fm.flex_rgba[fid] = HOT if hot else COOL
        lit_v += int(hot)

    # ---- report, then shoot ------------------------------------------------- #
    import collections
    cb = collections.Counter()
    for i in range(rd.ncon):
        c = rd.contact[i]
        for g in (c.geom1, c.geom2):
            n = mj.mj_id2name(rm, mj.mjtObj.mjOBJ_BODY, rm.geom_bodyid[g]) or ""
            if "cube" not in n.lower():
                cb[n] += 1
    fired_bodies = {layout.entries[i].body for i in np.nonzero(live)[0]}
    print(f"[rigid] contacts by body: {dict(cb)}")
    print(f"[rigid] taxels fired: {int(live.sum())}/368 on bodies {sorted(fired_bodies)}")
    print(f"[flex ] patches loaded: {lit_v}/{fm.nflex}")
    silent = [b for b in cb if b and b not in fired_bodies]
    print(f"[gate ] bodies WITH contact but ZERO taxels firing: {silent}")

    # NOTE: the side-by-side against the flex hand is NOT rendered here any more.
    # Measured: driven by this grasp the flex hand DROPS the cube (it travels
    # from [0.11, 0, 0.10] to [-0.17, -0.15, -0.21]) and 39 of its 43 contacts
    # are skin-on-skin rather than skin-on-cube. Putting the two panels together
    # would show a hand gripping an object beside a hand that has let go, and
    # invite a comparison neither supports. The flex side needs a grasp tuned to
    # hold, which is a separate piece of work.
    #
    # What IS rendered is the finding that stands on the rigid side alone, and
    # does not depend on flex at all: every fingertip touches the cube, and not
    # one of its taxels fires.
    _shoot(rm, rd, OUT / "A_fingertips_silent.png", az=138, el=-38, dist=0.40)
    print(f"wrote {OUT / 'A_fingertips_silent.png'}  (rigid hand, taxels green=fired)")

    # Close-up on the index fingertip: contact disc vs the pads it misses.
    tip = mj.mj_name2id(rm, mj.mjtObj.mjOBJ_BODY, "if_ds")
    _shoot(rm, rd, OUT / "B_cause.png", az=150, el=-22, dist=0.055,
           lookat=rd.xpos[tip].copy())
    print(f"wrote {OUT / 'B_cause.png'}  (index fingertip close-up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
