"""Inject the 368 taxel sites into the model mjlab trains on, then check them.

This is the prototype of the real mjlab change. mjlab builds its hand through
`MjSpec` (robots/leap_xela.py:get_spec), so doing it here with MjSpec means the
code transfers almost verbatim into an mjlab spec_fn -- no ElementTree text
surgery like the original scene_builder.py needed for MuJoCo 3.1.1.

Two checks, because "it compiled" proves nothing:

1. NUMERIC: every taxel should sit just off the skin of its own body. For each
   site, measure the distance to the nearest mesh vertex of the body it was
   attached to. Sites on a body whose frames genuinely agree land within a few
   mm of the surface. Sites on a mismatched body land far away, or buried.

2. VISUAL: render them. A taxel map that is numerically fine but visually wrong
   (mirrored patch, rotated pad, sites inside the mesh) is caught here and
   nowhere else.

Run:
    PYTHONPATH=third_party/leapXELA_model .venv/bin/python explore/03_inject_taxels.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import pathlib

import mujoco as mj
import numpy as np
from PIL import Image

from leapxela.taxel_layout import build_layout
from taxel_map import BODY_MAP, CONFIRMED, RL_XML, SUSPECT, body_shapes

OUT = pathlib.Path(__file__).resolve().parent / "out"

SITE_SIZE = 0.0012
GREEN = [0.1, 0.9, 0.2, 1.0]   # body frames confirmed
RED = [0.95, 0.15, 0.1, 1.0]   # fingertip, geometry mismatch

# VirtualTaxelSensor's patch-locality cutoff. Defaults from
# leapxela/visualize_taxel_layout.py: --kernel-sigma 0.0035, --kernel-cutoff 0.01.
KERNEL_CUTOFF_M = 0.01


def burial_depths(model: mj.MjModel, layout) -> dict[str, list[float]]:
    """How far is solid geometry still outside each taxel, along its own normal?

    Ray-cast from the site along its +z (outward) normal. A hit means material
    remains outside the taxel, i.e. it sits inside the link. The trace is
    restricted to the taxel's OWN body -- otherwise, in the rest pose, rays
    strike neighbouring fingers and the numbers are meaningless.
    """
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    original = model.geom_group.copy()
    mask = np.array([1, 0, 0, 0, 0, 0], np.uint8)   # trace group 0 only
    gid = np.zeros(1, np.int32)
    out: dict[str, list[float]] = {}
    try:
        for e in layout.entries:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, BODY_MAP[e.body])
            model.geom_group[:] = 1
            model.geom_group[model.geom_bodyid == bid] = 0
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, e.site_name)
            p = data.site_xpos[sid].copy()
            n = data.site_xmat[sid].reshape(3, 3)[:, 2].copy()
            dist = mj.mj_ray(model, data, p, n, mask, 1, -1, gid)
            out.setdefault(e.body, []).append(dist if dist >= 0 else np.inf)
    finally:
        model.geom_group[:] = original
    return out


def inject(spec: mj.MjSpec, layout) -> tuple[int, int]:
    """Add one site per taxel to the mapped body. Returns (confirmed, suspect)."""
    bodies = {b.name: b for b in spec.bodies}
    n_ok = n_sus = 0
    for e in layout.entries:
        target = BODY_MAP[e.body]
        body = bodies.get(target)
        if body is None:
            raise KeyError(f"body '{target}' not in model (mapped from '{e.body}')")
        suspect = e.body in SUSPECT
        body.add_site(
            name=e.site_name,
            pos=np.asarray(e.pos, dtype=float),
            quat=np.asarray(e.quat, dtype=float),
            size=[SITE_SIZE] * 3,
            group=4,
            rgba=RED if suspect else GREEN,
        )
        n_sus += suspect
        n_ok += not suspect
    return n_ok, n_sus


def main() -> None:
    layout = build_layout()

    print(f"target: {RL_XML.relative_to(pathlib.Path.cwd())}")
    spec = mj.MjSpec.from_file(str(RL_XML))
    existing = [s.name for s in spec.sites if s.name.startswith("taxel_")]
    if existing:
        raise SystemExit(
            f"model already has {len(existing)} taxel_* sites -- this is the flex "
            "model, not the rigid one the splat encoder needs. See PROJECT_LOG §1.5b."
        )
    print(f"  rigid hand: {len(spec.bodies) - 1} bodies, no taxel sites (as expected)")

    n_ok, n_sus = inject(spec, layout)
    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    print(f"injected {n_ok + n_sus} taxel sites  ({n_ok} confirmed, {n_sus} suspect)")
    print(f"model now has {model.nsite} sites total\n")

    # ---- check 1: distance from each taxel to its body's skin ---------------
    shapes = body_shapes(model, keep_verts=True)
    print("=" * 78)
    print("DISTANCE FROM EACH TAXEL TO THE SURFACE OF ITS OWN BODY")
    print("=" * 78)
    print(f"{'tactile body':<28}{'->':^4}{'RL body':<9}{'n':>4}"
          f"{'median':>9}{'p95':>9}{'max':>9}   status")
    print("-" * 78)

    worst_confirmed = 0.0
    for src, dst in BODY_MAP.items():
        idx = [i for i, e in enumerate(layout.entries) if e.body == src]
        verts = shapes[dst].verts
        dists = []
        for i in idx:
            site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, layout.entries[i].site_name)
            # site pos is stored in body-local coords, same frame as verts
            p = model.site_pos[site_id]
            dists.append(np.linalg.norm(verts - p, axis=1).min())
        d = np.array(dists) * 1000.0
        tag = "CONFIRMED" if src in CONFIRMED else "suspect"
        if src in CONFIRMED:
            worst_confirmed = max(worst_confirmed, float(d.max()))
        print(f"{src:<28}{'->':^4}{dst:<9}{len(idx):>4}"
              f"{np.median(d):>9.2f}{np.percentile(d, 95):>9.2f}{d.max():>9.2f}   {tag}")

    print("-" * 78)
    print(f"worst distance-to-skin among CONFIRMED bodies: {worst_confirmed:.2f} mm\n")

    # ---- check 1b: burial depth vs the encoder's patch-locality gate --------
    # VirtualTaxelSensor does: weights[abs(local_z) > kernel_cutoff] = 0.
    # local_z is the contact's offset along the taxel normal, so a taxel buried
    # d mm below the surface sees local_z ~ d for a contact directly above it.
    # If d >= kernel_cutoff that taxel can never fire. THIS is the criterion --
    # not whether the taxel is visually on the surface. In the reference model
    # 248/368 taxels are inside their own link mesh, so burial is normal.
    print("=" * 78)
    print("BURIAL DEPTH vs kernel_cutoff  (the gate that actually matters)")
    print("=" * 78)
    depths = burial_depths(model, layout)
    ok = 0
    for src in BODY_MAP:
        v = np.array(depths[src])
        fin = np.isfinite(v)
        med = np.median(v[fin]) * 1000 if fin.any() else 0.0
        mx = v[fin].max() * 1000 if fin.any() else 0.0
        safe = mx < KERNEL_CUTOFF_M * 1000
        ok += safe
        print(f"  {src:<28} buried {fin.sum():>3}/{len(v):<3}"
              f" median {med:>5.2f} mm  max {mx:>5.2f} mm   "
              f"{'OK' if safe else 'EXCEEDS CUTOFF — taxel can never fire'}")
    print("-" * 78)
    print(f"bodies within the {KERNEL_CUTOFF_M * 1000:.0f} mm cutoff: {ok}/{len(BODY_MAP)}")
    print("(default kernel_cutoff=0.01, kernel_sigma=0.0035 -- visualize_taxel_layout.py)\n")

    # ---- check 2: look at them ---------------------------------------------
    OUT.mkdir(exist_ok=True)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 1200
    # The pinned file is the bare hand, with no scene: no lights, no skybox.
    # Brighten the headlight or every render comes out near-black.
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    model.vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]

    opt = mj.MjvOption()
    opt.sitegroup[4] = 1          # taxel sites live in group 4
    opt.geomgroup[3] = 0          # hide collision-only geoms

    # Frame the model automatically, then orbit around that.
    base = mj.MjvCamera()
    mj.mjv_defaultFreeCamera(model, base)

    views = {
        "taxels_palm": dict(azimuth=90, elevation=-89, zoom=0.62),
        "taxels_side": dict(azimuth=35, elevation=-20, zoom=0.62),
        "taxels_tip": dict(azimuth=115, elevation=-40, zoom=0.28),
    }
    with mj.Renderer(model, height=1200, width=1600) as r:
        for name, view in views.items():
            c = mj.MjvCamera()
            c.type = mj.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = base.lookat
            c.distance = base.distance * view["zoom"]
            c.azimuth, c.elevation = view["azimuth"], view["elevation"]
            r.update_scene(data, camera=c, scene_option=opt)
            path = OUT / f"{name}.png"
            Image.fromarray(r.render()).save(path)
            print(f"  wrote {path}")

    print("\ngreen = frames confirmed (232 taxels), red = fingertip mismatch (136)")


if __name__ == "__main__":
    main()
