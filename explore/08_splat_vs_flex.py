"""Do the splat encoder and the flex skin report the same thing?

WHY THIS MATTERS. The project measured four tactile representations and found a
hard split: the flex/bubble skin is the physically richest one and is ~217x too
slow to run at RL scale (H4), while the splat encoder is hardware-faithful and,
with the Triton kernel, costs ~1% (H2). The supervisor's own tactile dataset
pipeline (`sparsh-skin-sim/data_collection`) records FLEX quantities -- flex
vertex displacement and Kelvin-Voigt force -- not splat output.

That sets up a question nobody has answered: **can the splat encoder stand in for
the flex skin?** If it can, it is the only way to get flex-like tactile signal at
training speed, because flex itself cannot run there. If it cannot, that is a
calibration gap worth knowing about before anything is built on top of either.

WHAT MAKES THE COMPARISON POSSIBLE. The two sensor models turn out to line up
structurally, which was not obvious in advance:

  * both describe exactly 368 taxels
  * the flex model has 18 flexes whose vertex counts match the splat layout's 18
    patches exactly -- 3 x 24 (uspa46 palm), 11 x 16 (uspa44 links), 4 x 30 (tips)
  * both hands expose the SAME 16 actuators under the SAME names, so a grasp
    command is literally the same vector for both

WHAT MAKES IT HARD, AND WHAT THIS SCRIPT DOES ABOUT IT.

  1. FLAT ORDERING DIFFERS. The splat layout's flat index is XELA hardware ID
     order, which interleaves patches (thumb tip, a 4x4, another 4x4, back to
     thumb tip). Flex vertices are grouped patch by patch. Mapping one to the
     other needs `sparsh-skin-sim/util/fk_taxel_util.py`'s patch tables, grid
     flips and 180-degree rotations, plus a taxel-map JSON and a URDF.
     **This script does not do that mapping**, deliberately: getting it subtly
     wrong yields a comparison that looks fine and means nothing, which is a
     failure mode this project has already hit twice. Everything below is
     computed PER PATCH and is therefore invariant to within-patch ordering.
     Per-taxel correlation is a separate job, and needs the mapping validated on
     its own first.

  2. THE HANDS ARE DIFFERENT GEOMETRY. Rigid RL-lineage vs flex tactile-lineage,
     ~10 mm difference at the fingertips (D4). So this is not a numerical
     equivalence test and must not be reported as one. It asks the weaker,
     answerable question: driven by the same grasp, do the two sensors agree
     about WHICH patches are loaded, WHEN, and roughly HOW HARD?

  3. THE FLEX SCENE HAS NOTHING TO GRIP. `scene_flex_sensor_Box.xml` contains no
     object. The cube and floor are injected from `scene_mjx_cube.xml` and the
     flex hand's welded root is moved onto the rigid hand's palm pose, exactly as
     `benchmarks/harness.py:build_touch_model` does for the touch model.

OUTPUT: `explore/out/splat_vs_flex.npz` plus a printed per-patch table.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import mujoco as mj
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "explore" / "out"
for _p in (ROOT / "benchmarks", ROOT / "explore", ROOT / "explore" / "vendor",
           ROOT / "third_party" / "leapXELA_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FLEX_SCENE = ROOT / "third_party" / "leapXELA_model" / "scene_flex_sensor_Box.xml"
PREGRIP_S = 3.0       # simulated seconds held at pregrip before pattern time
SETTLE_S = 2.0        # the grasp patterns' shared close prefix
RUN_S = 6.0           # modulated phase to record
# Specified in simulated SECONDS, never in steps: a fixed step count silently
# means different amounts of settling once the timestep changes, which is how the
# first corrected run ended up reporting zero contacts.
ACTIVE_N = 1e-3       # a taxel counts as "live" above this force magnitude


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Patch correspondence
# --------------------------------------------------------------------------- #


def _add_cube_only(spec: mj.MjSpec, ref: mj.MjModel) -> None:
    """Give `spec` the reorientation cube from `scene_mjx_cube.xml`.

    `harness._add_cube_and_floor` adds a floor too, which the flex scene already
    has -- MjSpec rejects the duplicate name. Only the cube is needed here: it is
    what the hand grips, and every parameter is copied off the COMPILED reference
    model so the two scenes cannot drift on size, mass, friction or contact
    regime.
    """
    cube_b = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_BODY, "cube")
    cube_g = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_GEOM, "cube")

    body = spec.worldbody.add_body()
    body.name = "cube"
    body.pos = ref.body_pos[cube_b].copy()
    body.quat = ref.body_quat[cube_b].copy()
    body.add_freejoint().name = "cube_freejoint"
    g = body.add_geom()
    g.name = "cube"
    g.type = mj.mjtGeom.mjGEOM_BOX
    g.size = ref.geom_size[cube_g].copy()
    g.mass = float(ref.body_mass[cube_b])
    g.friction = ref.geom_friction[cube_g].copy()
    g.condim = int(ref.geom_condim[cube_g])
    g.contype = int(ref.geom_contype[cube_g])
    g.conaffinity = int(ref.geom_conaffinity[cube_g])
    g.group = 3


def flex_patch_centroids(model: mj.MjModel, data: mj.MjData) -> dict[str, tuple]:
    """Rest-pose world centroid and vertex count for every flex, from the model.

    Flexcomp attaches vertices to bodies it generates itself (`flex_uspa46_1_0`,
    `flex_if_tip_1`), NOT to the hand links, so the flex-to-hand-link
    correspondence cannot be read off body names. Position can be, and is what
    the matching below uses.
    """
    out = {}
    for fid in range(model.nflex):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, fid)
        adr = int(model.flex_vertadr[fid])
        num = int(model.flex_vertnum[fid])
        verts = data.flexvert_xpos[adr:adr + num]
        out[name] = (verts.mean(axis=0).copy(), num)
    return out


def splat_patch_centroids(model: mj.MjModel, data: mj.MjData, layout):
    """Rest-pose world centroid, count and flat indices for every splat patch."""
    groups: dict[tuple, list[int]] = {}
    for i, e in enumerate(layout.entries):
        groups.setdefault((e.body, e.patch), []).append(i)
    site_ids = np.array(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, e.site_name)
         for e in layout.entries]
    )
    out = {}
    for key, idx in groups.items():
        idx = np.asarray(idx, dtype=int)
        pos = data.site_xpos[site_ids[idx]]
        out[key] = (pos.mean(axis=0).copy(), len(idx), idx)
    return out


def match_by_geometry(flex_c: dict, splat_c: dict, max_mm: float = 40.0):
    """Pair flex patches to splat patches by rest-pose proximity, size-constrained.

    Name-based matching would be guesswork here: the splat patch ids (`4_4_1`,
    `4_6_2`) carry an ordering convention that is not documented anywhere in the
    layout, and three of the flex patches (`*_bs_uspa44`) sit on hand-link bases
    that `BODY_MAP` fuses into `palm`, so body name alone cannot separate them.
    Position can, because both models describe the same physical hand and the
    flex hand's welded root has been moved onto the rigid hand's palm pose.

    Greedy nearest-neighbour, restricted to equal taxel counts, one-to-one, and
    reported with the residual distance so a bad pairing is visible rather than
    silent. Anything beyond `max_mm` is refused rather than accepted.
    """
    pairs, used = [], set()
    order = sorted(flex_c.items(), key=lambda kv: kv[0])
    for fname, (fpos, fnum) in order:
        best, bestd = None, np.inf
        for key, (spos, snum, idx) in splat_c.items():
            if key in used or snum != fnum:
                continue
            d = float(np.linalg.norm(fpos - spos))
            if d < bestd:
                best, bestd = (key, idx), d
        if best is None or bestd * 1000.0 > max_mm:
            print(f"[warn ] {fname} ({fnum} verts): no acceptable splat patch "
                  f"(nearest {bestd*1000:.1f} mm) -- skipped")
            continue
        used.add(best[0])
        pairs.append((fname, best[0], best[1], bestd * 1000.0))
    return pairs


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-mm", type=float, default=None,
                    help="override VirtualTaxelSensor's kernel_cutoff (default "
                         "10 mm, the reference constant). DIAGNOSTIC ONLY -- "
                         "changing it means the encoder no longer reproduces the "
                         "reference (D5), so nothing measured with it may be "
                         "reported as splat-vs-flex agreement without saying so.")
    args = ap.parse_args()

    import harness as H
    gm = _load(ROOT / "explore" / "05_grasp_motions.py", "grasp_motions")
    import flex_util
    from leapxela.taxel_layout import build_layout
    from leapxela.touch_sensor import VirtualTaxelSensor
    from taxel_map import BODY_MAP, remap_layout

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- flex model, with the rigid scene's cube put in front of it -------- #
    ref = mj.MjModel.from_xml_path(str(H.DEFAULT_SCENE))
    spec = mj.MjSpec.from_file(str(FLEX_SCENE))
    bodies = {b.name: b for b in spec.bodies}
    palm_ref = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_BODY, "palm")
    bodies["palm"].pos = ref.body_pos[palm_ref].copy()
    bodies["palm"].quat = ref.body_quat[palm_ref].copy()
    _add_cube_only(spec, ref)
    fm = spec.compile()
    # The flex model's OWN timestep (0.002 s, 50 solver iterations). Forcing the
    # rigid sweeps' 0.01 s here makes it diverge -- MuJoCo reports
    # "Nan, Inf or huge value in QACC" and the Kelvin-Voigt readout returns ~1e5 N.
    # An earlier version of this script did that and the resulting comparison was
    # void. The RIGID model is then stepped at the same dt: a smaller timestep is
    # always stable for it, and matching the two keeps frames in 1:1
    # correspondence with no substepping and no resampling.
    dt = float(fm.opt.timestep)
    fd = mj.MjData(fm)
    mj.mj_forward(fm, fd)
    print(f"[flex ] nbody={fm.nbody} nflex={fm.nflex} nu={fm.nu} "
          f"dt={dt} iters={fm.opt.iterations} "
          f"cube injected: {mj.mj_name2id(fm, mj.mjtObj.mjOBJ_BODY, 'cube') >= 0}")

    # ---- rigid model with taxel sites + the reference splat encoder --------- #
    rm = H.build_scene_model(H.DEFAULT_SCENE, "splat")
    rm.opt.timestep = dt          # match the flex model, not env_cfg
    rd = mj.MjData(rm)
    mj.mj_forward(rm, rd)
    layout = remap_layout(build_layout())
    # Same kernel constants the oracle fixture was banked with (harness.py).
    cutoff = H.KERNEL_CUTOFF if args.cutoff_mm is None else args.cutoff_mm / 1000.0
    splat = VirtualTaxelSensor(rm, layout, H.cube_geom_names(rm),
                               H.KERNEL_SIGMA, cutoff)
    if args.cutoff_mm is not None:
        print(f"[warn ] kernel_cutoff overridden: {H.KERNEL_CUTOFF*1000:.1f} -> "
              f"{cutoff*1000:.1f} mm. This is NOT the reference sensor any more.")
    print(f"[splat] nbody={rm.nbody} nu={rm.nu} taxels={len(layout.entries)}")

    # ---- patch correspondence, by rest-pose geometry ---------------------- #
    mj.mj_forward(fm, fd)
    mj.mj_forward(rm, rd)
    flex_c = flex_patch_centroids(fm, fd)
    splat_c = splat_patch_centroids(rm, rd, layout)
    pairs = match_by_geometry(flex_c, splat_c)
    print(f"[match] {len(pairs)}/{len(flex_c)} patches paired "
          f"({sum(len(p[2]) for p in pairs)} taxels); "
          f"residuals {min(p[3] for p in pairs):.1f}-{max(p[3] for p in pairs):.1f} mm"
          if pairs else "[match] none")
    for fname, skey, idx, d in sorted(pairs, key=lambda p: -p[3])[:4]:
        print(f"        worst: {fname:22s} -> {skey[0]}/{skey[1]:10s} {d:5.1f} mm")
    if not pairs:
        print("[fail ] no patch correspondence; cannot compare")
        return 2

    # ---- drive both hands with the same grasp patterns ---------------------- #
    n_settle = int(round(SETTLE_S / dt))
    n_run = int(round(RUN_S / dt))
    est = flex_util.AllFlexForceEstimator(fm)

    # Actuator index maps: same 16 names, but the models order them the same way
    # and only 2 of 16 share ctrl RANGES, which is the crux of the next comment.
    ridx = np.array([mj.mj_name2id(rm, mj.mjtObj.mjOBJ_ACTUATOR,
                                   mj.mj_id2name(fm, mj.mjtObj.mjOBJ_ACTUATOR, i))
                     for i in range(fm.nu)])
    r_lo, r_hi = rm.actuator_ctrlrange[:, 0], rm.actuator_ctrlrange[:, 1]

    def targets_for(t: float, profile):
        """Same JOINT ANGLES on both hands, not the same grasp fraction.

        `grasp_target` returns `lo + fraction * (hi - lo)`, and the two hands
        share only 2 of 16 actuator ranges -- the flex hand's lower limits are
        much tighter (e.g. `if_mcp` -0.087 vs -0.314). Driving both with the same
        FRACTION therefore puts them in different poses, which would make any
        sensor comparison meaningless: the hands would not be gripping the cube
        the same way.

        So the pattern is evaluated once on the flex model -- the lineage his
        `motion_util` patterns were written against -- and the resulting absolute
        angles are applied to both, clipped to whatever each model allows.

        This also removes the need for 05_grasp_motions' THUMB_OPPOSED_FRACTION
        deviation: that existed because his 0.35 thumb fraction maps to a more
        open pose on the RL hand's wider range. Passing absolute angles sidesteps
        the mismatch instead of retuning around it.
        """
        tf = gm.grasp_target(fm, t, profile)
        tr = np.clip(tf[np.argsort(ridx)] if False else tf, r_lo[ridx], r_hi[ridx])
        return tf, tr

    rows = []
    for pattern in gm.GRASP_PATTERNS:
        profile = gm.GraspProfile(pattern=pattern)
        mj.mj_resetData(fm, fd)
        mj.mj_resetData(rm, rd)
        est.reset()

        # Hold at pregrip so the cube settles before the pattern starts.
        tf0, tr0 = targets_for(0.0, profile)
        fd.ctrl[:] = tf0
        rd.ctrl[ridx] = tr0
        for _ in range(int(round(PREGRIP_S / dt))):
            mj.mj_step(fm, fd)
            mj.mj_step(rm, rd)
            est.update(fm, fd)

        # TARE. The Kelvin-Voigt estimate is never exactly zero: every flex
        # vertex carries some displacement and velocity even untouched, so an
        # absolute threshold marks all 368 "active" in every frame and the
        # activity column becomes meaningless (it read 16.00/24.00/30.00
        # everywhere in the first attempt). Recording the resting value at the
        # end of pregrip and subtracting it is exactly what taring a real
        # tactile sensor does, and it is what makes the two readouts comparable
        # at all -- the splat encoder is zero by construction when nothing is
        # touching.
        base = {k: v.copy() for k, v in est.update(fm, fd).items()}

        for k in range(n_settle + n_run):
            t = k * dt
            tf, tr = targets_for(t, profile)
            fd.ctrl[:] = tf
            rd.ctrl[ridx] = tr
            mj.mj_step(fm, fd)
            mj.mj_step(rm, rd)
            raw = est.update(fm, fd)
            fforce = {k2: raw[k2] - base[k2] for k2 in raw}   # tared
            if k < n_settle:
                continue
            s = splat.update(rd)                        # (368, 3)
            for fname, skey, sidx, _d in pairs:
                fv = fforce[fname]                      # (n_vert, 3)
                fmag = np.linalg.norm(fv, axis=1)
                smag = np.linalg.norm(s[sidx], axis=1)
                rows.append((
                    pattern, k, fname, f"{skey[0]}/{skey[1]}",
                    float(fmag.sum()), float(smag.sum()),
                    int((fmag > ACTIVE_N).sum()), int((smag > ACTIVE_N).sum()),
                    float(fmag.max()), float(smag.max()),
                ))
        print(f"[run  ] {pattern:12s} flex peak {est.peak_force_magnitude(fforce):8.3f} N   "
              f"splat peak {np.linalg.norm(s, axis=1).max():8.3f} N   "
              f"ncon rigid {rd.ncon:3d} flex {fd.ncon:3d}")

    # ---- per-patch summary --------------------------------------------------- #
    import collections
    agg = collections.defaultdict(lambda: [[], [], [], []])
    for (_, _, fname, sname, fsum, ssum, fact, sact, fpk, spk) in rows:
        a = agg[(fname, sname)]
        a[0].append(fsum); a[1].append(ssum); a[2].append(fact); a[3].append(sact)

    print(f"\n{'flex patch':24s} {'splat patch':22s} "
          f"{'flexN':>8} {'splatN':>8} {'ratio':>7} {'flexAct':>8} {'splatAct':>9} {'corr':>7}")
    print("-" * 100)
    corrs = []
    for (fname, sname), (fs, ss, fa, sa) in sorted(agg.items()):
        fs, ss = np.asarray(fs), np.asarray(ss)
        c = (np.corrcoef(fs, ss)[0, 1]
             if fs.std() > 1e-12 and ss.std() > 1e-12 else np.nan)
        if np.isfinite(c):
            corrs.append(c)
        print(f"{fname:24s} {sname:22s} {fs.mean():8.3f} {ss.mean():8.3f} "
              f"{(ss.mean()/fs.mean() if fs.mean() else np.nan):7.2f} "
              f"{np.mean(fa):8.2f} {np.mean(sa):9.2f} {c:7.3f}")

    print(f"\nmean per-patch correlation over time: "
          f"{np.nanmean(corrs):.3f}  (n={len(corrs)} patches)")

    np.savez_compressed(
        OUT / "splat_vs_flex.npz",
        rows=np.array(rows, dtype=object),
        patterns=np.array(gm.GRASP_PATTERNS),
        pairs=np.array([(a, str(b), d) for a, b, _, d in pairs], dtype=object),
    )
    print(f"wrote {OUT / 'splat_vs_flex.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
