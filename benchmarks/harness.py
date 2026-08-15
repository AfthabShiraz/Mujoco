"""Measure MuJoCo Warp throughput and memory for ONE env count, then exit.

Deliberately single-shot: it measures one N and terminates. `scale_sweep.py`
runs it once per N in a separate memory-capped process, so an out-of-memory at
high N kills a child instead of wedging the machine (PROJECT_LOG.md §1.1 --
this box has unified memory, so an oversized run takes down the host, not just
the process).

Timing rules, because getting these wrong makes the numbers meaningless:
  * warm up first -- kernel compilation and allocation dominate the first steps
  * `wp.synchronize()` before starting and before stopping the clock; GPU work
    is asynchronous, so timing without it measures queue-submission speed
  * report env-steps/sec (N x steps / wall), the scale-invariant number

Three representations:

  --tactile none    bare physics. The H1 control. Unchanged from the original
                    harness, so previously banked numbers stay comparable.

  --tactile splat   physics + the naive Gaussian-splat taxel encoder
                    (`src/encoders/taxel_torch.py`) evaluated every step for
                    every world. This is what H2 is about: at what env count
                    does touch cost more than the physics it observes?

  --tactile touch   physics + MuJoCo's own 421 `<touch>` sensors, which Warp
                    evaluates INSIDE `mjw.step`. No encoder of ours is
                    involved; see the timing note below for how its cost is
                    separated from physics, and `build_touch_model` for why the
                    sensors are injected into the splat scene rather than the
                    supervisor's standalone touch model being used directly.

The splat path is measured TWICE per run, on purpose:
  * a fused loop with no inner synchronisation -> the honest end-to-end
    throughput, directly comparable to the `none` numbers;
  * a second loop that synchronises between `mjw.step` and the encoder ->
    the physics/encoder split. Those inner syncs serialise the pipeline and
    cost a little, which is why they are not used for the headline figure.
    `split_ms_per_step` is reported alongside so the overhead is visible.

The touch path CANNOT be measured that way, and the difference matters:
`mujoco_warp` computes touch sensors from inside `mjw.step` (`sensor.sensor_acc`
-> `_sensor_touch`, dispatched over `dim=(naconmax, n_touch_sensors)`), so there
is no separate stage to put a clock around. Reporting 0.0 there would be a lie,
and reporting the whole step as "encoder" would be a different lie. Instead the
touch path runs the SAME loop several times over with pieces switched off at the
host level, and reports differences:

    P_head     step + readout, all sensors on        -> headline throughput
    P_full     step only,      all sensors on
    P_notouch  step only,      sensors on but `m.sensor_touch_adr` swapped for a
               size-0 array, which makes the `_sensor_touch` launch a no-op and
               changes nothing else in `sensor_acc`
    P_nosens   step only,      `DisableBit.SENSOR` -> the whole sensor stage skipped
    P_read     readout only

    physics_ms      = P_notouch                <- the MATCHED baseline
    encoder_ms      = P_head     - P_notouch   <- touch kernel + readout, the
                                                  field comparable to splat's
    touch_kernel_ms = P_full     - P_notouch   <- the naconmax x 421 dispatch
    sensor_stage_ms = P_notouch  - P_nosens    <- mujoco_warp's fixed per-step
                                                  cost for having ANY sensor

`P_notouch`, not `P_nosens`, is the baseline, and the difference is not
cosmetic. `scene_mjx_cube.xml` already declares 29 `framepos`/`framequat`
sensors, so every previously banked sweep -- physics-only and both splat runs --
already pays mujoco_warp's whole sensor stage, including the `weld_geom_list`
buffer `sensor_acc` allocates and fills on every single call whether or not any
tactile sensor exists. Charging that to touch would inflate its cost by a fixed
amount it does not cause. `P_nosens` is still measured, so that overhead is
reported (`sensor_stage_ms_per_step`) rather than hidden -- it is not small.

Every one of those loops is fused (syncs only at the boundaries), so unlike the
splat split there is no pipeline-drain penalty and `physics + encoder` equals the
measured total exactly. The price is that each figure is a difference of two
wall-clock measurements, so small ones sit in the noise; `timing_noise_ms_per_step`
is a repeat of `P_nosens` and gives the floor below which they mean nothing.

Everything in the timed loop stays on the GPU. Warp arrays are viewed as torch
tensors through `wp.to_torch` (zero copy, same memory); nothing calls `.numpy()`
between the clocks. In particular `d.nacon` -- the live contact count -- is
never read to the host: it is broadcast as a device tensor to build the live
mask. A host round-trip there would cost a full device sync per step and would
dominate the very measurement this file exists to make.

Emits one JSON object on stdout. Any human-readable commentary goes to stderr.

Run directly (prefer scale_sweep.py):
    .venv/bin/python benchmarks/harness.py --num-envs 64
    .venv/bin/python benchmarks/harness.py --num-envs 4 --tactile splat --verify
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The tactile path needs the encoder (src), BODY_MAP (explore) and the taxel
# layout + reference sensor (third_party/leapXELA_model). Set here rather than
# demanded as PYTHONPATH so `scale_sweep.py` can spawn children with no
# environment fixup.
for _extra in (ROOT / "src", ROOT / "explore", ROOT / "third_party" / "leapXELA_model"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

ASSETS = (
    ROOT / "third_party" / "leapXelaMjLab" / "src" / "leap_xela_mjlab"
    / "assets" / "leapXELA_model"
)
DEFAULT_SCENE = ASSETS / "scene_mjx_cube.xml"

# The supervisor's touch-sensor hand: 421 `<touch>` sensors on 421 sites.
# Used as the *donor* of the sensor layout, and -- with `--touch-scene
# supervisor` -- as a whole model, to reproduce the zero-reading result
# documented in `build_touch_model`.
TOUCH_MODEL = ASSETS / "robot_touch_sensor_array_mjx_generated_model.xml"

# The supervisor's binary-touchgrid hand. Donor of the 300 `*_sensor_*` patch
# geoms -- 5 mm and 2.5 mm boxes, `contype=1 conaffinity=1 condim=3` -- spread
# over 13 bodies (palm 144, px/md/th links 16 each, four fingertips 7 each).
#
# NOTE for the write-up: the plan doc calls this "361 collision geoms". 361 is
# the count of `<geom ` LINES in the file, which includes visuals, the coarse
# collision boxes and the class defaults. The number of geoms that actually
# constitute the sensor grid is **300**, measured off the compiled model. The
# argument is unaffected -- 300 extra colliding boxes against `nconmax=48` is
# still the H3 setup -- but the figure that gets quoted should be the right one.
TOUCHGRID_MODEL = ASSETS / "robot_touch_sensor_array_binary_touchgrid_generated_mjx.xml"

# Matches leapXelaMjLab env_cfg.py; the static contact budget is part of what
# we are measuring, so it must not drift.
#
# CAREFUL -- these two are scoped differently in mujoco_warp's put_data, and
# getting it wrong silently destroys the measurement:
#   naconmax  is a TOTAL across worlds  -> multiply by nworld (default: 48*nworld)
#   njmax     is PER WORLD              -> do NOT multiply (default: 64)
# Passing njmax=120*nworld makes each world's constraint space nworld times too
# large; the dense solver (JTDAJ + Cholesky) then costs O(N^2) and throughput
# *falls* as N rises. Measured: N=512 went 337 -> 67,840 env-steps/s on fixing it.
NCONMAX_PER_ENV = 48
NJMAX_PER_ENV = 120

# The hand must actually be gripping, or the benchmark times a contact-free
# scene and tells you nothing about contact cost.
# Match leapXelaMjLab's env_cfg.py: sim_dt=0.01 (ctrl_dt=0.05, decimation=5).
# The scene we load includes leapXela_generated_mjx.xml, which declares
# timestep=0.001 -- 10x smaller than what the supervisor actually trains at.
# Left uncorrected, our env-steps/s would not be comparable to his training
# throughput (10x more steps for the same simulated robot time).
SIM_DT = 0.01

# The rest of leapXelaMjLab's SimulationCfg / MujocoCfg (env_cfg.py:329-341).
# These are NOT set by this harness -- they are inherited from the scene XML,
# and today they happen to agree. `assert_solver_matches_supervisor` below is
# what turns that coincidence into a checked invariant.
#
# Worth being paranoid about, because this exact failure has already happened
# once: the scene declares timestep=0.001 while he trains at 0.01, and every
# sweep taken before that was found silently measured a 10x-finer simulation
# (HYPOTHESES.md D9). A change to `iterations` on either side would be the same
# bug with no warning -- throughput would move and the cause would look like
# anything but a config drift.
SUPERVISOR_OPT = {
    "integrator": mj.mjtIntegrator.mjINT_EULER,
    "iterations": 5,
    "ls_iterations": 8,
}

CLOSE_FRACTION = 0.6
SETTLE_STEPS = 250   # CPU run shows first cube contact ~step 120

# VirtualTaxelSensor defaults, from leapxela/visualize_taxel_layout.py. The
# oracle in explore/out/oracle_fixture.npz was banked with these.
KERNEL_SIGMA = 0.0035
KERNEL_CUTOFF = 0.01
SITE_SIZE = 0.0012


def assert_solver_matches_supervisor(mjm: mj.MjModel, strict: bool = True) -> dict:
    """Check the compiled model's solver settings against leapXelaMjLab's config.

    Reports every field either way so the run record carries what it actually
    ran with, rather than what it was assumed to run with.
    """
    got = {
        "integrator": int(mjm.opt.integrator),
        "iterations": int(mjm.opt.iterations),
        "ls_iterations": int(mjm.opt.ls_iterations),
    }
    mismatched = {k: (v, int(SUPERVISOR_OPT[k])) for k, v in got.items()
                  if v != int(SUPERVISOR_OPT[k])}
    if mismatched and strict:
        detail = ", ".join(f"{k}: model={g} supervisor={w}"
                           for k, (g, w) in mismatched.items())
        raise SystemExit(
            f"[harness] REFUSING TO RUN: solver settings differ from "
            f"leapXelaMjLab env_cfg.py -- {detail}. Throughput measured under "
            f"different solver settings is not comparable to his training, and "
            f"the difference would not otherwise be visible. Fix the model, or "
            f"pass --allow-solver-drift and say so in the write-up."
        )
    return {f"opt_{k}": v for k, v in got.items()} | {"solver_drift": bool(mismatched)}


def peak_rss_gb() -> float:
    """Peak resident set of this process. On unified memory this tracks the
    GPU allocation too, which is exactly the number we care about here."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def touch_sensor_donor() -> tuple[mj.MjModel, np.ndarray]:
    """The supervisor's touch model and the ids of its 421 `<touch>` sensors."""
    src = mj.MjModel.from_xml_path(str(TOUCH_MODEL))
    ids = np.nonzero(src.sensor_type == mj.mjtSensor.mjSENS_TOUCH)[0]
    return src, ids


def _add_cube_and_floor(spec: mj.MjSpec, ref: mj.MjModel) -> None:
    """Give `spec` the reorientation cube and floor from `scene_mjx_cube.xml`.

    Every parameter is copied off the COMPILED reference model rather than
    retyped from the XML, so the two scenes cannot drift: same size, mass,
    friction, condim and contype/conaffinity means the same contact regime.

    Two things from the reference scene are deliberately left out, and neither
    touches the dynamics: the cube's textured visual mesh (`contype=0`,
    `conaffinity=0`, `density=0`) and the mocap `goal` body (both its geoms are
    `contype=0 conaffinity=0`). Skipping the mesh also avoids fighting the touch
    model's `meshdir="./assets/"`, which does not point at `./meshes/`.
    """
    cube_b = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_BODY, "cube")
    cube_g = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_GEOM, "cube")
    floor_g = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_GEOM, "floor")

    # contype/conaffinity 2/2 on the floor: it catches a dropped cube but is
    # invisible to the hand, exactly as in the reference scene.
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mj.mjtGeom.mjGEOM_PLANE
    floor.pos = ref.geom_pos[floor_g].copy()
    floor.size = ref.geom_size[floor_g].copy()
    floor.contype = int(ref.geom_contype[floor_g])
    floor.conaffinity = int(ref.geom_conaffinity[floor_g])

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


def build_touch_model(scene: pathlib.Path, variant: str) -> mj.MjModel:
    """Compile a scene carrying the supervisor's 421 native `<touch>` sensors.

    `variant="supervisor"` is the obvious thing to do and it does not work.
    It pairs `robot_touch_sensor_array_mjx_generated_model.xml` with this scene's
    cube (translating its palm onto the RL model's palm pose, so the hand meets
    the cube exactly as in `scene_mjx_cube.xml`). The grasp is fine -- 8 live
    hand/cube contacts at 2.4-6.8 N -- and **all 421 sensors read exactly zero,
    on GPU and on CPU alike.**

    Measured cause, not a guess: a `<touch>` sensor sums the normal force of
    contacts whose contact point lies INSIDE its site. These sites are spheres of
    radius 1.0 mm. That model's collision set was simplified for MJX -- the
    uspa44/uspa46 sensor pads the taxels sit on were reduced to visual meshes,
    leaving 19 colliding geoms against the RL model's 36 -- so the cube only ever
    touches the coarse `*_collision_*` boxes underneath. Nearest touch site to
    any contact point at a gripping pose: **9.42 mm**, against a 1.00 mm zone.
    No grip force closes a 9 mm geometric gap. (`robot_touch_sensor_array_gem.xml`,
    the same array with 92 colliding geoms, was checked too: 11.86 mm, also zero.)

    `variant="inject"` (the default, and what the sweep uses) therefore does for
    touch exactly what `--tactile splat` does for the splat layout: it takes the
    421 sites out of the supervisor's model and injects them, with their
    `<touch>` sensors, into the RL scene the other three sweeps already use. In
    that model the sensor pads ARE collision geoms, so contacts land on them --
    284 of the 337 sites that share a body with a collision box sit within their
    own 1 mm zone of that box's surface -- and the sensors fire.

    That choice is what makes the comparison fair, and it is worth being explicit
    about why: `inject` changes NOTHING about the physics relative to
    `scale_sweep_physics_only_dt01.csv` and `scale_sweep_splat_dt01.csv` beyond
    421 massless, collisionless sites. Same bodies, same collision set, same
    contacts. The only new thing in the step is the sensor evaluation, which is
    the thing being measured.
    """
    ref = mj.MjModel.from_xml_path(str(scene))
    src, touch_ids = touch_sensor_donor()

    if variant == "supervisor":
        spec = mj.MjSpec.from_file(str(TOUCH_MODEL))
        bodies = {b.name: b for b in spec.bodies}
        palm = mj.mj_name2id(ref, mj.mjtObj.mjOBJ_BODY, "palm")
        # The touch model parks its hand 136 mm from where the RL model puts it.
        # The palm is the root and is welded to the world, so overwriting its
        # pose is a rigid move of the whole hand: it aligns hand-to-cube geometry
        # with the reference scene without perturbing the kinematic chain.
        bodies["palm"].pos = ref.body_pos[palm].copy()
        bodies["palm"].quat = ref.body_quat[palm].copy()
        _add_cube_and_floor(spec, ref)
        return spec.compile()

    if variant != "inject":
        raise ValueError(f"unknown touch scene variant {variant!r}")

    spec = mj.MjSpec.from_file(str(scene))
    bodies = {b.name: b for b in spec.bodies}
    for sid in touch_ids:
        site = int(src.sensor_objid[sid])
        bname = mj.mj_id2name(src, mj.mjtObj.mjOBJ_BODY, int(src.site_bodyid[site]))
        if bname not in bodies:
            raise KeyError(f"body '{bname}' carries a touch site but is not in {scene.name}")
        # "tt_" because the two lineages share several site names (uspa46_1 ...)
        # and MjSpec rejects duplicates. Body-local pos/quat/size/type are copied
        # verbatim -- the sensor zone must not be resized, it is what decides
        # whether a contact registers.
        name = "tt_" + mj.mj_id2name(src, mj.mjtObj.mjOBJ_SITE, site)
        s = bodies[bname].add_site()
        s.name = name
        s.pos = src.site_pos[site].copy()
        s.quat = src.site_quat[site].copy()
        s.size = src.site_size[site].copy()
        s.type = mj.mjtGeom(int(src.site_type[site]))
        s.group = 4
        sn = spec.add_sensor()
        sn.name = name + "_touch"
        sn.type = mj.mjtSensor.mjSENS_TOUCH
        sn.objtype = mj.mjtObj.mjOBJ_SITE
        sn.objname = name
    return spec.compile()


def touchgrid_donor() -> tuple[mj.MjModel, np.ndarray]:
    """The supervisor's touchgrid model and the ids of its 300 sensor-patch geoms.

    Selected by name (`*_sensor_*`) rather than by MuJoCo class, because the
    class tree splits the patches across `sensor_patch` and seven
    `fingertip_sensor_*_surface_*` subclasses and the compiled model does not
    retain which default a geom came from. The name convention is uniform across
    all 300 and is what the supervisor's own tooling keys on.
    """
    src = mj.MjModel.from_xml_path(str(TOUCHGRID_MODEL))
    ids = np.array(
        [g for g in range(src.ngeom)
         if "_sensor_" in (mj.mj_id2name(src, mj.mjtObj.mjOBJ_GEOM, g) or "")],
        dtype=int,
    )
    return src, ids


def build_touchgrid_model(scene: pathlib.Path, collide: str) -> mj.MjModel:
    """Compile the RL scene with the supervisor's 300 touchgrid patch geoms in it.

    Same injection strategy as `--tactile touch`, and for the same reason: it
    holds every other variable fixed against the already-banked physics-only and
    splat sweeps, so a difference in the numbers is caused by the touchgrid and
    not by a different hand, a different cube or a different contact regime.

    The difference from the touch path is the whole point of H3. Touch sites are
    massless and collisionless -- they observe the physics. Touchgrid patches
    **are collision geoms**: they *change* the physics, taking the scene from 71
    geoms to 371 and generating contacts of their own against a contact budget
    (`nconmax=48` per world) that is static and was not raised to accommodate
    them. Whether that shows up as slowdown, as silently dropped contacts, or as
    both, is the hypothesis.

    Three variants, because one number here would be misleading:

    `off`  -- the same 300 patches with `contype=conaffinity=0`. Geometrically and
        kinematically the identical model (same bodies, same 371 geoms, same
        forward kinematics, same mass) that generates no contacts of its own.
        Measured on CPU it reproduces the bare scene's contact counts exactly
        (ncon 24, nefc 112), which is what makes it a valid control. It isolates
        the cost of *having* the geoms from the cost of the contacts they make.

    `naive` -- the patches exactly as the donor declares them, `contype=1
        conaffinity=1`, free to collide with the hand's own links and each other.

    `object` -- `contype=2 conaffinity=0`, which passes the mask against the cube
        (`contype=1 conaffinity=2`) and against nothing else: not the hand's
        collision boxes (`1/1`), not other patches. A taxel is supposed to sense
        the object rather than its own knuckles, so this is the modelling choice
        a real deployment would make, and it is the default here.

        The floor (`contype=2 conaffinity=2`) does technically pass the mask
        against a `2/0` patch. The hand is welded well above it and never reaches
        it -- but `TouchgridReadout.contact_mix` counts patch-floor contacts every
        run rather than trusting that argument.

    **Measured, and the mask turns out not to matter:** `naive` and `object` give
    identical contact counts in this scene, peaking at **ncon 232 / nefc 944**
    against the bare scene's 24 / 112 -- and the breakdown at that peak is 223
    patch-cube, 0 patch-patch, 0 patch-hand, 0 patch-floor. The patches never
    touch anything but the cube, so freeing them to self-collide changes nothing.

    That makes the explosion **intrinsic to the representation**, not an artifact
    of injecting into a different model lineage: 300 small boxes tiling the
    hand's surface against a cube generate ~223 simultaneous contacts where the
    hand's 37 coarse collision boxes generate ~9. Against the supervisor's
    `nconmax=48` / `njmax=120` that is a 5x contact and 8x constraint shortfall,
    and mujoco_warp does not degrade gracefully when it hits it: the narrowphase
    runs past its buffers and dies with an illegal memory access
    (`ccd_kernel`, `_qfrc_constraint_from_grad`). See H3.

    (An early version of this comment blamed the model-lineage mismatch of D4,
    on the grounds that the donor model peaks at only 38 contacts. That control
    was invalid -- the donor model standing alone has no cube in it, so it was
    measuring self-collision against nothing to grip.)

    The patches are forced massless. In the donor they inherit the default
    density and would add mass and inertia to the hand's links, which would
    change the dynamics for a reason that has nothing to do with sensing and
    would make the comparison against the other sweeps invalid.
    """
    if collide not in ("off", "naive", "object"):
        raise ValueError(f"unknown touchgrid collide mode {collide!r}")
    src, gids = touchgrid_donor()
    if gids.size == 0:
        raise RuntimeError(f"no `*_sensor_*` geoms found in {TOUCHGRID_MODEL.name}")

    spec = mj.MjSpec.from_file(str(scene))
    bodies = {b.name: b for b in spec.bodies}
    for gid in gids:
        gid = int(gid)
        bname = mj.mj_id2name(src, mj.mjtObj.mjOBJ_BODY, int(src.geom_bodyid[gid]))
        if bname not in bodies:
            raise KeyError(f"body '{bname}' carries a sensor patch but is not in {scene.name}")
        # "tg_" prefix for the same reason "tt_" exists on the touch path: the
        # two lineages share geom names and MjSpec rejects duplicates.
        g = bodies[bname].add_geom()
        g.name = "tg_" + mj.mj_id2name(src, mj.mjtObj.mjOBJ_GEOM, gid)
        g.type = mj.mjtGeom(int(src.geom_type[gid]))
        # Body-local pose and size copied verbatim off the compiled donor. A
        # resized patch is a different sensor: the box half-extents are what
        # decide which contacts it catches and how many contacts it makes.
        g.pos = src.geom_pos[gid].copy()
        g.quat = src.geom_quat[gid].copy()
        g.size = src.geom_size[gid].copy()
        g.condim = int(src.geom_condim[gid])
        g.friction = src.geom_friction[gid].copy()
        g.solref = src.geom_solref[gid].copy()
        g.solimp = src.geom_solimp[gid].copy()
        g.margin = float(src.geom_margin[gid])
        g.gap = float(src.geom_gap[gid])
        if collide == "off":
            g.contype = g.conaffinity = 0
        elif collide == "naive":
            g.contype = int(src.geom_contype[gid])
            g.conaffinity = int(src.geom_conaffinity[gid])
        else:  # "object" -- see the docstring for the mask arithmetic
            g.contype, g.conaffinity = 2, 0
        g.mass = 0.0
        g.group = 4
    return spec.compile()


def build_scene_model(scene: pathlib.Path, tactile: str,
                      touch_scene: str = "inject",
                      touchgrid_collide: str = "object") -> mj.MjModel:
    """Compile the benchmark scene, with taxel sites for the tactile paths.

    `none` compiles the scene untouched so its numbers stay comparable with the
    already-banked physics-only sweep. `splat` injects the 368 taxel sites via
    MjSpec exactly as `explore/03_inject_taxels.py` does -- the sites must exist
    in the model handed to `mjw.put_model`, because that is what allocates
    `d.site_xpos` / `d.site_xmat`. `touch` is handled by `build_touch_model`.

    Sites are massless and collisionless, so they do not change the dynamics;
    they do add 368 site poses to the forward kinematics, and that cost lands in
    the *physics* half of the split. That is the honest place for it -- it is a
    cost the tactile representation imposes.
    """
    if tactile == "none":
        return mj.MjModel.from_xml_path(str(scene))
    if tactile == "touch":
        return build_touch_model(scene, touch_scene)
    if tactile == "touchgrid":
        return build_touchgrid_model(scene, touchgrid_collide)

    from leapxela.taxel_layout import build_layout
    from taxel_map import BODY_MAP

    spec = mj.MjSpec.from_file(str(scene))
    bodies = {b.name: b for b in spec.bodies}
    for e in build_layout().entries:
        target = BODY_MAP[e.body]
        if target not in bodies:
            raise KeyError(f"body '{target}' not in model (mapped from '{e.body}')")
        bodies[target].add_site(
            name=e.site_name,
            pos=np.asarray(e.pos, dtype=float),
            quat=np.asarray(e.quat, dtype=float),
            size=[SITE_SIZE] * 3,
            group=4,
        )
    return spec.compile()


def cube_geom_names(model: mj.MjModel) -> list[str]:
    """Named geoms on the manipulated cube (excluding the mocap goal).

    Same rule as `explore/04_run_encoder.py`, which produced the oracle. Do not
    let the two diverge: the object/hand test is what sets the force sign.
    """
    names = []
    for gid in range(model.ngeom):
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if "cube" in body.lower() and "goal" not in body.lower() and gname:
            names.append(gname)
    return names


# --------------------------------------------------------------------------- #
def _prefix_layout(layout, prefix: str):
    """Copy `layout` with `prefix` prepended to every body and site name.

    For mjlab, which namespaces scene elements by entity (`robot/palm`,
    `robot/taxel_017`). Kept separate from `taxel_map.remap_layout` because the
    two fix different things and compose: remap converts tactile-lineage body
    names to RL-lineage ones, this converts RL-lineage names to mjlab's
    namespaced ones.
    """
    import dataclasses

    # `TaxelEntry` is frozen, so rebuild rather than mutate -- same construction
    # `taxel_map.remap_layout` uses, and for the same reason: the layout is part
    # of the correctness oracle's input and must not be edited in place.
    entries = tuple(
        dataclasses.replace(e, body=f"{prefix}{e.body}",
                            site_name=f"{prefix}{e.site_name}")
        for e in layout.entries
    )
    return dataclasses.replace(layout, entries=entries)


def _resolve_encoder(name: str):
    """Pick the encoder implementation. All three must be numerically equivalent;
    `tests/test_triton_kernel.py` is what guarantees that, not this function."""
    from encoders.taxel_torch import encode_taxels

    if name == "torch":
        return encode_taxels
    if name == "compile":
        # one-line fusion win: removes DRAM round trips, changes no arithmetic
        import torch
        return torch.compile(encode_taxels, dynamic=False)
    if name == "triton":
        import sys
        sys.path.insert(0, str(ROOT / "src" / "kernels" / "triton"))
        from taxel_triton import encode_taxels_triton
        return encode_taxels_triton
    raise ValueError(f"unknown encoder {name!r}")


# The splat path
# --------------------------------------------------------------------------- #


class SplatEncoder:
    """Warp contacts -> padded (B, C, ...) batch -> `encode_taxels`, all on GPU.

    The hard part is not the encoder, it is the layout mismatch. MuJoCo Warp
    keeps contacts in ONE flat array of length `naconmax` shared by every world,
    tagged with `contact.worldid`, in whatever order collision detection
    happened to emit them (verified: not grouped by world). The encoder wants a
    dense per-env `(B, C, ...)` batch. So every step has to bucket the flat
    array by world and rank contacts within their bucket.

    That bucketing is done with a stable sort by worldid rather than with atomic
    counters, for one reason: determinism. Slot assignment is a permutation, so
    either way the *set* of contacts per world is the same -- but the encoder
    sums over contacts in slot order, and float addition is not associative, so
    atomics would make the output vary run to run at the 1e-7 level. That is
    exactly the kind of wobble that makes an oracle comparison unfalsifiable
    (HYPOTHESES.md H6). The sort is over `naconmax` (~48N) elements; the encoder
    behind it is over B*C*T (~72M at N=4096), so it is not where the time goes.

    C is `NCONMAX_PER_ENV` (48) -- the per-env contact budget mjlab configures.
    Contacts beyond the 48th in a world are dropped; `dropped_contacts()`
    reports whether that ever happened, because silently discarding contacts
    would be a correctness bug wearing a performance disguise.
    """

    def __init__(self, mjm: mj.MjModel, m, d, nworld: int, encoder: str = "torch",
                 name_prefix: str = "", object_geom_names: list[str] | None = None,
                 contacts_per_env: int | None = None):
        import torch
        import warp as wp

        from encoders.taxel_torch import N_TAXELS, prepare_static
        from leapxela.taxel_layout import build_layout
        from taxel_map import remap_layout

        self.torch = torch
        self.wp = wp
        self.m, self.d = m, d
        self.B = nworld
        # Defaults to the supervisor's budget, but mjlab is free to configure a
        # different `nconmax` and the bucketing width must follow the model that
        # is actually loaded, not a constant.
        self.C = contacts_per_env if contacts_per_env is not None else NCONMAX_PER_ENV
        self.T = N_TAXELS
        self.naconmax = int(d.naconmax)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev

        # Layout body names are tactile-lineage; the model is RL-lineage. Skip
        # the remap and every lookup returns -1, all 368 taxels collapse into
        # one bogus group and the encoder outputs zeros without erroring
        # (PROJECT_LOG §1.5d). This is the single easiest way to get a fast
        # wrong answer here.
        #
        # `name_prefix` is the second half of that same trap. mjlab namespaces
        # every element by its entity -- bodies become `robot/palm`, the cube
        # geom becomes `cube/cube` -- so a layout carrying bare names misses
        # everything there too. `prepare_static` raises rather than zeroing, so
        # this fails loudly, but only if the prefix is actually threaded through.
        self._encode_fn = _resolve_encoder(encoder)
        layout = remap_layout(build_layout())
        if name_prefix:
            layout = _prefix_layout(layout, name_prefix)
        if object_geom_names is None:
            object_geom_names = cube_geom_names(mjm)
        self.static = prepare_static(mjm, layout, object_geom_names)

        self.site_ids = torch.as_tensor(self.static.site_ids, dtype=torch.long, device=dev)
        self.taxel_body = torch.as_tensor(self.static.taxel_body, dtype=torch.long, device=dev)
        self.geom_body_slot = torch.as_tensor(
            self.static.geom_body_slot, dtype=torch.long, device=dev
        )
        self.geom_is_object = torch.as_tensor(
            self.static.geom_is_object, dtype=torch.bool, device=dev
        )
        self.ngeom = int(mjm.ngeom)

        # ---- zero-copy views of the Warp arrays ---------------------------- #
        # These alias the Warp allocations, so they stay valid for the life of
        # `d` and cost nothing per step. Nothing below ever leaves the device.
        self.v_pos = wp.to_torch(d.contact.pos)          # (naconmax, 3)
        self.v_frame = wp.to_torch(d.contact.frame)      # (naconmax, 3, 3)
        self.v_geom = wp.to_torch(d.contact.geom)        # (naconmax, 2) int32
        self.v_worldid = wp.to_torch(d.contact.worldid)  # (naconmax,)  int32
        self.v_nacon = wp.to_torch(d.nacon)              # (1,)         int32
        self.v_site_xpos = wp.to_torch(d.site_xpos)      # (B, nsite, 3)
        self.v_site_xmat = wp.to_torch(d.site_xmat)      # (B, nsite, 3, 3)

        # `contact_force` writes here; ask for every slot and mask the dead ones
        # rather than reading `nacon` back to the host to size the launch.
        self.force_ids = wp.array(
            np.arange(self.naconmax, dtype=np.int32), dtype=wp.int32
        )
        self.force_buf = wp.zeros(self.naconmax, dtype=wp.spatial_vectorf)
        self.v_force = wp.to_torch(self.force_buf)       # (naconmax, 6)

        # ---- preallocated scratch ------------------------------------------ #
        n, BC = self.naconmax, self.B * self.C
        self.idx = torch.arange(n, dtype=torch.long, device=dev)
        self.world_range = torch.arange(self.B + 1, dtype=torch.long, device=dev)
        self.trash = torch.full((n,), BC, dtype=torch.long, device=dev)  # bin for dead slots

        # +1 row is the trash bin every dead/overflowing contact is written to,
        # then sliced away. Keeps every shape static: no boolean-mask indexing,
        # which would need a device->host sync to size its output.
        self.b_pos = torch.zeros((BC + 1, 3), dtype=torch.float32, device=dev)
        self.b_frame = torch.zeros((BC + 1, 3, 3), dtype=torch.float32, device=dev)
        self.b_force = torch.zeros((BC + 1, 6), dtype=torch.float32, device=dev)
        self.b_sign = torch.zeros((BC + 1,), dtype=torch.float32, device=dev)
        self.b_body = torch.zeros((BC + 1,), dtype=torch.long, device=dev)
        self.b_valid = torch.zeros((BC + 1,), dtype=torch.bool, device=dev)

        self.last = None   # (B, T, 3), most recent encoding

    # -- contact bucketing ---------------------------------------------------- #

    def _batch(self):
        """Flat Warp contacts -> the padded (B, C, ...) tensors the encoder wants."""
        torch = self.torch
        B, C, BC = self.B, self.C, self.B * self.C

        # Live mask without touching the host: `nacon` is a (1,) device tensor.
        live = self.idx < self.v_nacon.to(torch.long)

        # Geom ids are -1 for flex contacts and garbage in the dead tail, so
        # clamp before indexing -- a -1 would wrap round to the last geom.
        g1 = self.v_geom[:, 0].to(torch.long).clamp_(0, self.ngeom - 1)
        g2 = self.v_geom[:, 1].to(torch.long).clamp_(0, self.ngeom - 1)

        # Vectorised copy of `contact_inputs_from_geoms`, kept structurally
        # identical to the numpy version it mirrors (including the asymmetry:
        # the geom1-is-object branch wins ties, and fixes the sign).
        hand_is_g2 = self.geom_is_object[g1] & (self.geom_body_slot[g2] >= 0)
        hand_is_g1 = ~hand_is_g2 & self.geom_is_object[g2] & (self.geom_body_slot[g1] >= 0)
        valid = (hand_is_g2 | hand_is_g1) & live
        body = torch.where(hand_is_g2, self.geom_body_slot[g2], self.geom_body_slot[g1])
        body = torch.where(valid, body, torch.full_like(body, -1))
        sign = torch.where(hand_is_g2, 1.0, -1.0).to(torch.float32)

        # Bucket by world. Dead contacts get key B so they sort to the end and
        # never claim a slot; the sort is stable so slot order is reproducible.
        wid = torch.where(live, self.v_worldid.to(torch.long), self.world_range[B])
        order = torch.argsort(wid, stable=True)
        wid_s = wid[order]
        # wid_s is sorted, so the first index of world w is just a binary search
        # -- an exclusive prefix sum of the per-world counts, without the
        # host sync that torch.bincount's output sizing would cost.
        starts = torch.searchsorted(wid_s, self.world_range)
        rank = self.idx - starts[wid_s]
        keep = (wid_s < B) & (rank < C)
        dest = torch.where(keep, wid_s * C + rank, self.trash)

        self._keep, self._live = keep, live   # diagnostics, read outside the clock

        # Zero first: slots no live contact maps to must not keep last step's
        # values. `index_copy_` duplicates only ever collide in the trash row.
        for buf in (self.b_pos, self.b_frame, self.b_force, self.b_sign, self.b_valid):
            buf.zero_()
        self.b_body.fill_(-1)

        self.b_pos.index_copy_(0, dest, self.v_pos[order])
        self.b_frame.index_copy_(0, dest, self.v_frame[order])
        self.b_force.index_copy_(0, dest, self.v_force[order])
        self.b_sign.index_copy_(0, dest, sign[order])
        self.b_body.index_copy_(0, dest, body[order])
        self.b_valid.index_copy_(0, dest, valid[order])

        return (
            self.b_pos[:BC].view(B, C, 3),
            self.b_frame[:BC].view(B, C, 3, 3),
            self.b_force[:BC].view(B, C, 6),
            self.b_sign[:BC].view(B, C),
            self.b_body[:BC].view(B, C),
            self.b_valid[:BC].view(B, C),
        )

    def encode(self):
        """One full tactile readout: (B, 368, 3) taxel forces, on the GPU."""
        encode_taxels = self._encode_fn

        # `to_world_frame=False` gives the wrench in the CONTACT frame, which is
        # what `mj_contactForce` returns and what the reference encoder consumes
        # before rotating by `contact.frame.T`. encode_taxels does that rotation
        # itself (einsum "bcji,bcj->bci"), so handing it the contact-frame force
        # plus contact.frame reproduces the reference arithmetic exactly. Asking
        # Warp for the world frame instead would double-rotate.
        self.mjw.contact_force(self.m, self.d, self.force_ids, False, self.force_buf)

        pos, frame, force, sign, body, valid = self._batch()
        self.last = encode_taxels(
            contact_pos=pos,
            contact_frame=frame,
            contact_force=force,
            contact_sign=sign,
            contact_body=body,
            contact_valid=valid,
            site_pos=self.v_site_xpos[:, self.site_ids],
            site_mat=self.v_site_xmat[:, self.site_ids],
            taxel_body=self.taxel_body,
            kernel_sigma=KERNEL_SIGMA,
            kernel_cutoff=KERNEL_CUTOFF,
        )
        return self.last

    # -- diagnostics (host reads; call OUTSIDE the timed loop) ---------------- #

    def dropped_contacts(self) -> int:
        """Live object<->hand contacts that did not fit in the per-env budget."""
        relevant = (self._live & (self.b_valid[:0].numel() == 0)) if False else None
        del relevant
        kept = int(self._keep.sum().item())
        live = int(self._live.sum().item())
        return live - kept

    def signal_stats(self) -> tuple[int, float]:
        """(active taxels in world 0, peak |force| over the batch).

        A GPU tactile path that returns all zeros is the documented failure mode
        here (PROJECT_LOG §1.5d), and it is indistinguishable from a fast
        correct one on a stopwatch. Every run prints this.
        """
        mag = self.last.norm(dim=-1)
        return int((mag[0] > 1e-6).sum().item()), float(mag.max().item())


def _attach_mjw(enc: SplatEncoder, mjw) -> SplatEncoder:
    enc.mjw = mjw
    return enc


# --------------------------------------------------------------------------- #
# The native-touch path
# --------------------------------------------------------------------------- #


class TouchReadout:
    """Read `d.sensordata` -> a dense (B, 421) tensor of touch normal forces.

    Deliberately thin. Unlike `SplatEncoder` there is nothing to compute here:
    `mjw.step` has already run `_sensor_touch` and written every value into
    `d.sensordata`. All that is left is to lift the 421 touch columns out of the
    450-column sensor block into the contiguous batch a policy would consume.

    That gather is `index_select`, not a slice, even though in this model the
    touch addresses happen to be contiguous (16..436 for the supervisor model,
    an appended block for the injected one -- `contiguous_adr` records which).
    A slice would be a zero-copy view, which would make the readout free by
    construction and hide the one real cost on this path: B x 421 x 4 bytes of
    traffic per step. `index_select` is what a general sensor layout would need
    and it materialises the tensor, so the number means something.

    It also holds the two host-side switches the timing split uses. Both are
    plain Python attributes on the Warp `Model` -- `sensor_acc` reads them on
    every call to decide what to launch -- so flipping them costs one branch and
    changes no allocation, no state and no dynamics.
    """

    def __init__(self, mjm: mj.MjModel, m, d, nworld: int, mjw):
        import torch
        import warp as wp

        self.torch, self.wp, self.mjw = torch, wp, mjw
        self.m, self.d, self.B = m, d, nworld

        ids = np.nonzero(mjm.sensor_type == mj.mjtSensor.mjSENS_TOUCH)[0]
        if ids.size == 0:
            raise RuntimeError(f"no <touch> sensors in the compiled model")
        adr = mjm.sensor_adr[ids].astype(np.int64)
        self.n_sensors = int(ids.size)
        self.contiguous_adr = bool(np.all(np.diff(adr) == 1))
        self.sensor_names = [mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_SENSOR, int(i)) for i in ids]

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adr = torch.as_tensor(adr, dtype=torch.long, device=dev)
        self.v_sensordata = wp.to_torch(d.sensordata)   # (B, nsensordata), zero copy
        self.last = None

        # The two gates. `sensor_touch_adr` is only ever read to size and feed
        # the `_sensor_touch` launch (mujoco_warp/_src/sensor.py:2514), so
        # substituting a size-0 array turns that one launch into a no-op and
        # leaves the rest of `sensor_acc` -- including its per-step
        # `weld_geom_list` allocation -- exactly as it was. That is what isolates
        # the 421-sensor dispatch from mujoco_warp's fixed sensor-stage cost.
        self._touch_adr = m.sensor_touch_adr
        self._empty_adr = wp.zeros(0, dtype=int)
        self._disableflags = int(m.opt.disableflags)

    # -- host-side switches, flipped between timed loops --------------------- #

    def set_touch(self, on: bool) -> None:
        self.m.sensor_touch_adr = self._touch_adr if on else self._empty_adr

    def set_sensors(self, on: bool) -> None:
        from mujoco_warp._src.types import DisableBit
        if on:
            self.m.opt.disableflags = self._disableflags
        else:
            self.m.opt.disableflags = self._disableflags | int(DisableBit.SENSOR)

    def read(self):
        """One tactile readout: (B, 421) normal forces, on the GPU."""
        self.last = self.torch.index_select(self.v_sensordata, 1, self.adr)
        return self.last

    # -- diagnostics (host reads; call OUTSIDE the timed loop) --------------- #

    def signal_stats(self) -> dict:
        """Proof the sensors are not all zero.

        This is the same trap `SplatEncoder.signal_stats` guards: an array that
        always reads zero benchmarks at exactly the same speed as one that
        works. For touch the trap is sharper, because the sensor zone is a
        1 mm sphere and a contact 2 mm away registers nothing at all --
        see `build_touch_model` for the variant where this is what happens.
        """
        t = self.last.detach().cpu().numpy()
        nz = t > 0.0
        return dict(
            n_touch_sensors=self.n_sensors,
            touch_contiguous_adr=self.contiguous_adr,
            touch_active_world0=int(nz[0].sum()),
            touch_active_any_world=int(nz.any(axis=0).sum()),
            touch_active_mean_per_world=float(nz.sum(axis=1).mean()),
            touch_min_nonzero_n=float(t[nz].min()) if nz.any() else 0.0,
            touch_max_n=float(t.max()),
            touch_total_n_world0=float(t[0].sum()),
        )


class TouchgridReadout:
    """Read the contact list -> a dense (B, 300) binary patch-contact map.

    This is the whole sensor. A binary touchgrid does not measure force: a patch
    reads 1 if any contact this step involves it and 0 otherwise, so the readout
    is a scatter of ones from the contact list into a per-world patch vector.
    MuJoCo computes nothing for it -- unlike `<touch>`, there is no sensor stage
    to gate -- which is exactly why H3 says its cost lives *inside* the physics
    step rather than in the readout.

    Static shapes throughout, and no device->host sync. Dead contact slots
    (`slot >= nacon`) and contacts between two non-patch geoms are aimed at a
    trash row that is sliced off, which is the same trick `SplatEncoder` uses and
    for the same reason: boolean-mask indexing would have to read `nacon` back to
    the host to size its output, and that sync inside the timed loop would
    contaminate the measurement it is there to take.

    Both columns of `d.contact.geom` are scattered, not just one. A patch-vs-cube
    contact has the patch in one column and the cube in the other, and which one
    depends on geom ordering; a patch-vs-patch contact (two patches on adjacent
    links touching) must light up both. Aiming non-patch geoms at the trash row
    is what makes scattering both columns correct.
    """

    def __init__(self, mjm: mj.MjModel, m, d, nworld: int):
        import torch
        import warp as wp

        self.torch, self.wp = torch, wp
        self.m, self.d, self.B = m, d, nworld
        self.naconmax = int(d.naconmax)

        names = [mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(mjm.ngeom)]
        patch = [g for g, nm in enumerate(names) if nm.startswith("tg_")]
        if not patch:
            raise RuntimeError("no `tg_` patch geoms in the compiled model")
        self.S = len(patch)
        self.patch_ids = patch
        self.ngeom = int(mjm.ngeom)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # geom id -> patch slot, or -1 for every geom that is not a patch.
        slot = np.full(mjm.ngeom, -1, dtype=np.int64)
        slot[np.asarray(patch, dtype=np.int64)] = np.arange(self.S, dtype=np.int64)
        self.geom_slot = torch.as_tensor(slot, dtype=torch.long, device=dev)

        # ---- zero-copy views of the Warp contact arrays --------------------- #
        self.v_geom = wp.to_torch(d.contact.geom)        # (naconmax, 2) int32
        self.v_worldid = wp.to_torch(d.contact.worldid)  # (naconmax,)   int32
        self.v_nacon = wp.to_torch(d.nacon)              # (1,)          int32

        self.idx = torch.arange(self.naconmax, dtype=torch.long, device=dev)
        self.out = torch.zeros((self.B * self.S + 1,), dtype=torch.float32, device=dev)
        self.trash = self.B * self.S
        self.last = None

    def read(self):
        """One tactile readout: (B, 300) binary patch contacts, on the GPU."""
        torch = self.torch
        live = self.idx < self.v_nacon.to(torch.long)         # (naconmax,)

        # Geom ids are -1 for flex contacts and garbage in the dead tail, and
        # `worldid` is garbage there too. Clamp both BEFORE indexing: an
        # out-of-range id is an illegal memory access, and -1 would silently
        # wrap to the last geom. `live` is what actually discards them.
        gid = self.v_geom.to(torch.long).clamp(0, self.ngeom - 1)   # (naconmax, 2)
        world = self.v_worldid.to(torch.long).clamp(0, self.B - 1)  # (naconmax,)
        slot = self.geom_slot[gid]                                  # (naconmax, 2)

        flat = world.unsqueeze(1) * self.S + slot             # (naconmax, 2)
        keep = live.unsqueeze(1) & (slot >= 0)
        flat = torch.where(keep, flat, self.trash)

        self.out.zero_()
        # Binary, so duplicate writes of 1.0 collide harmlessly -- the result
        # does not depend on scatter ordering and needs no atomics.
        self.out.scatter_(0, flat.reshape(-1), 1.0)
        self.last = self.out[:self.trash].view(self.B, self.S)
        return self.last

    # -- diagnostics (host reads; call OUTSIDE the timed loop) --------------- #

    def contact_mix(self, mjm: mj.MjModel) -> dict:
        """Break the live contact list down by what the patches are touching.

        This is what tells the three `--touchgrid-collide` modes apart in the
        data rather than in the argument: `object` should show patch-cube
        contacts and no patch-patch or patch-floor ones, `naive` shows patches
        hitting the hand's own links, `off` shows no patch contacts at all.
        """
        nacon = int(self.v_nacon.detach().cpu().numpy()[0])
        geom = self.v_geom.detach().cpu().numpy()[:nacon]
        is_patch = np.zeros(mjm.ngeom, dtype=bool)
        is_patch[np.asarray(self.patch_ids)] = True
        names = [mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_GEOM, g) or "" for g in range(mjm.ngeom)]
        is_floor = np.array([n == "floor" for n in names])
        is_cube = np.array([n == "cube" for n in names])

        g1, g2 = np.clip(geom[:, 0], 0, mjm.ngeom - 1), np.clip(geom[:, 1], 0, mjm.ngeom - 1)
        p1, p2 = is_patch[g1], is_patch[g2]
        any_patch = p1 | p2
        other = np.where(p1, g2, g1)
        return dict(
            touchgrid_contacts_live=nacon,
            touchgrid_patch_contacts=int(any_patch.sum()),
            touchgrid_patch_patch_contacts=int((p1 & p2).sum()),
            touchgrid_patch_cube_contacts=int((any_patch & is_cube[other]).sum()),
            touchgrid_patch_floor_contacts=int((any_patch & is_floor[other]).sum()),
            touchgrid_patch_hand_contacts=int(
                (any_patch & ~(p1 & p2) & ~is_cube[other] & ~is_floor[other]).sum()),
        )

    def signal_stats(self) -> dict:
        """Proof the grid is not reading all-zero -- the same trap as everywhere
        else here: a sensor that never fires benchmarks exactly like one that
        works."""
        t = self.last.detach().cpu().numpy()
        nz = t > 0.0
        return dict(
            n_patches=self.S,
            touchgrid_active_world0=int(nz[0].sum()),
            touchgrid_active_any_world=int(nz.any(axis=0).sum()),
            touchgrid_active_mean_per_world=float(nz.sum(axis=1).mean()),
        )


class GraspDriver:
    """Drives every world through the supervisor's own grasp patterns.

    WHY THIS EXISTS. Every throughput number banked before 2026-08-14 was taken
    at a single frozen pose: all 16 actuators driven to 60% of range once, 250
    settle steps, then time. That is a clean, reproducible operating point and it
    is also an unrepresentative one -- ~7.5 live contacts per env, constant, and
    *identical in every world*. A real rollout has contacts that move, vanish and
    reappear, and worlds that are decorrelated from one another.

    That matters unequally across the implementations being compared, which is
    exactly why it has to be fixed:

      * the dense torch encoder is contact-count blind. It computes
        `B x C x T` with C = nconmax = 48 whether 1 slot is live or all 48.
      * the Triton kernel loops the live contacts, so its cost tracks them:
        measured 0.54 ms at 1 contact/env, 0.57 at 4.3, 0.73 at 12, 1.27 at 48.

    So a quiet grasp flatters the kernel's *ratio* against the dense baseline.
    Both numbers were honest; the operating point was just soft. This driver
    replaces it with motion the supervisor actually designed.

    WHAT IT DRIVES. The seven patterns from `explore/05_grasp_motions.py`, which
    are themselves a verbatim port of `sparsh-skin-sim/util/motion_util.py`
    (`GRASP_PATTERNS`, `GraspProfile`, `grasp_target`) -- his constants, his
    shapes, his timings. Nothing is re-tuned here.

    WORLDS ARE DECORRELATED ON PURPOSE. World `w` gets pattern `w % 7` and a
    phase offset, so at any instant the batch holds a spread of grasp phases
    rather than one pose replicated N times. Two reasons, both about not
    measuring a fiction:
      * contact counts vary *across* the batch as well as over time, which is
        what a real training batch looks like;
      * it puts genuine load imbalance in front of the kernel. One block per
        (env, taxel-tile) with wildly different per-env contact counts is the
        case that exposes stragglers, and the frozen pose could never produce it.

    COST OF DRIVING. `ctrl` is rewritten every step from a precomputed GPU table
    -- one gather plus one copy, no host round trip, nothing recomputed inside
    the clock. It is charged to every variant equally, `none` included, so the
    comparison stays fair. `--motion static` reproduces the old operating point
    exactly, so the banked CSVs remain reproducible.
    """

    # Pattern time base. The shared prefix every pattern runs -- smoothstep close
    # from pregrip to grip over [0, 1.5] s with the thumb lagging 0.5 s -- ends at
    # 2.0 s, and only after that does the pattern-specific modulation begin. So
    # the settle phase plays [0, 2.0) and the timed phase cycles [2.0, 12.0),
    # which is the part that actually differs between patterns.
    CLOSE_END_S = 2.0
    CYCLE_END_S = 12.0

    def __init__(self, mjm: mj.MjModel, d, nworld: int, sim_dt: float):
        import torch
        import warp as wp

        sys.path.insert(0, str(ROOT / "explore"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "grasp_motions", ROOT / "explore" / "05_grasp_motions.py")
        gm = importlib.util.module_from_spec(spec)
        # Register before executing: `GraspProfile` is a dataclass, and
        # dataclasses resolve their field types through
        # `sys.modules[cls.__module__]`. Load the file without this and the
        # module is absent from `sys.modules`, so that lookup returns None and
        # the class fails to build.
        sys.modules[spec.name] = gm
        spec.loader.exec_module(gm)

        self.torch, self.B = torch, nworld
        self.patterns = list(gm.GRASP_PATTERNS)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_settle = int(round(self.CLOSE_END_S / sim_dt))
        n_cycle = int(round((self.CYCLE_END_S - self.CLOSE_END_S) / sim_dt))

        # Precompute each pattern's target vector on the host, once. `grasp_target`
        # is a pure function of time, so this is a table lookup thereafter and
        # nothing of his logic runs inside the timed loop.
        settle = np.empty((len(self.patterns), n_settle, mjm.nu), dtype=np.float32)
        cycle = np.empty((len(self.patterns), n_cycle, mjm.nu), dtype=np.float32)
        for p, name in enumerate(self.patterns):
            profile = gm.GraspProfile(pattern=name)
            for k in range(n_settle):
                settle[p, k] = gm.grasp_target(mjm, k * sim_dt, profile)
            for k in range(n_cycle):
                t = self.CLOSE_END_S + k * sim_dt
                cycle[p, k] = gm.grasp_target(mjm, t, profile)

        self.settle = torch.as_tensor(settle, device=dev)
        self.cycle = torch.as_tensor(cycle, device=dev)
        self.n_settle, self.n_cycle = n_settle, n_cycle

        w = torch.arange(nworld, device=dev)
        self.world_pattern = (w % len(self.patterns)).long()
        # Spread phase offsets evenly over the cycle so the batch covers the whole
        # modulation, rather than clustering wherever `w // 7` happens to land.
        groups = max(1, (nworld + len(self.patterns) - 1) // len(self.patterns))
        self.world_offset = (((w // len(self.patterns)) * n_cycle) // groups).long()

        self.v_ctrl = wp.to_torch(d.ctrl)      # (B, nu), zero copy
        self.k = 0

    def settle_step(self, k: int) -> None:
        """Play the shared close prefix. Every world closes on its own pattern."""
        idx = min(k, self.n_settle - 1)
        self.v_ctrl.copy_(self.settle[self.world_pattern, idx])

    def step(self) -> None:
        """One control update: gather each world's target for its own phase."""
        idx = (self.k + self.world_offset) % self.n_cycle
        self.v_ctrl.copy_(self.cycle[self.world_pattern, idx])
        self.k += 1

    def reset_phase(self) -> None:
        """Rewind the cycle so each timed loop sees the same motion.

        The touch path runs the same loop several times over and subtracts the
        results (see the module docstring). Those differences are only meaningful
        if every loop drives identical dynamics, so the phase must not carry over
        from one loop into the next.
        """
        self.k = 0


# --------------------------------------------------------------------------- #
# Correctness gate
# --------------------------------------------------------------------------- #


def verify_touchgrid(mjm, d, tg: TouchgridReadout) -> dict:
    """GPU patch map vs. CPU MuJoCo on world 0, from the same qpos.

    Cheaper and stricter than the splat gate: the quantity is a set of geom ids,
    so agreement is exact or it is not agreement. Reported rather than gated,
    because Warp and CPU MuJoCo do not have to produce byte-identical contact
    sets from the same state (plan §7) -- a small disagreement here is a contact
    discrepancy, not a readout bug, and the two must not be confused.
    """
    gpu = tg.read().detach().cpu().numpy()[0] > 0.0

    mjd = mj.MjData(mjm)
    mjd.qpos[:] = d.qpos.numpy()[0].astype(np.float64)
    mjd.qvel[:] = d.qvel.numpy()[0].astype(np.float64)
    mjd.ctrl[:] = d.ctrl.numpy()[0].astype(np.float64)
    mj.mj_forward(mjm, mjd)

    cpu = np.zeros(tg.S, dtype=bool)
    slot = {g: i for i, g in enumerate(tg.patch_ids)}
    for c in range(mjd.ncon):
        for g in (int(mjd.contact[c].geom1), int(mjd.contact[c].geom2)):
            if g in slot:
                cpu[slot[g]] = True

    c_world = d.contact.worldid.numpy()[:int(d.nacon.numpy()[0])]
    return dict(
        touchgrid_check_gpu_active=int(gpu.sum()),
        touchgrid_check_cpu_active=int(cpu.sum()),
        touchgrid_check_agree=int((gpu == cpu).sum()),
        touchgrid_check_disagree=int((gpu != cpu).sum()),
        touchgrid_check_gpu_ncon=int((c_world == 0).sum()),
        touchgrid_check_cpu_ncon=int(mjd.ncon),
    )


def verify(mjm, m, d, enc: SplatEncoder, tol: float = 1e-4) -> dict:
    """Check the GPU tactile path against the CPU reference before timing anything.

    Two independent checks, because they fail for different reasons:

    A. PLUMBING. Take the GPU's own contacts, rebuild the padded batch on the
       host with the *numpy* helpers (`contact_inputs_from_geoms` and the same
       padding `tests/test_encoder_oracle.py` uses), run `encode_taxels` on CPU,
       and compare. This isolates everything new in this file -- the worldid
       bucketing, the torch port of the object/hand test, the site gather, the
       contact-force frame convention -- from the encoder itself, which the
       oracle test already covers. This check is the gate: it should agree to
       float32 rounding, so a real failure means real broken plumbing.

    B. END TO END. Copy world 0's state into an `mj.MjData`, `mj_forward`, and
       run the supervisor's `VirtualTaxelSensor` on it. This is the only check
       that exercises `mjw.contact_force` against `mj_contactForce`. It is
       REPORTED, NOT GATED: the Warp solver runs in float32 with a different
       warm start and its own collision routines, so its contact set and
       constraint forces legitimately differ from the CPU's (plan §7, the
       "Warp-vs-CPU contact discrepancy"). Gating on it would be gating on
       MuJoCo, not on this code.
    """
    import torch

    from encoders.taxel_torch import contact_inputs_from_geoms, encode_taxels

    got = enc.encode().detach().cpu().numpy()
    B, C, T = enc.B, enc.C, enc.T

    # ---- A: same contacts, host-side assembly ------------------------------ #
    nacon = int(d.nacon.numpy()[0])
    c_pos = enc.v_pos.detach().cpu().numpy()[:nacon]
    c_frame = enc.v_frame.detach().cpu().numpy()[:nacon]
    c_force = enc.v_force.detach().cpu().numpy()[:nacon]
    c_geom = enc.v_geom.detach().cpu().numpy()[:nacon].astype(np.int64)
    c_world = enc.v_worldid.detach().cpu().numpy()[:nacon].astype(np.int64)
    body, sign, valid = contact_inputs_from_geoms(enc.static, c_geom[:, 0], c_geom[:, 1])

    p_pos = np.zeros((B, C, 3), np.float32)
    p_frame = np.zeros((B, C, 3, 3), np.float32)
    p_force = np.zeros((B, C, 6), np.float32)
    p_sign = np.zeros((B, C), np.float32)
    p_body = np.full((B, C), -1, np.int64)
    p_valid = np.zeros((B, C), bool)
    fill = np.zeros(B, np.int64)
    for i in range(nacon):                       # plain host loop; not timed
        w = int(c_world[i])
        k = fill[w]
        if k >= C:
            continue
        fill[w] += 1
        p_pos[w, k] = c_pos[i]
        p_frame[w, k] = c_frame[i]
        p_force[w, k] = c_force[i]
        p_sign[w, k] = sign[i]
        p_body[w, k] = body[i]
        p_valid[w, k] = valid[i]

    site_pos = enc.v_site_xpos[:, enc.site_ids].detach().cpu().numpy()
    site_mat = enc.v_site_xmat[:, enc.site_ids].detach().cpu().numpy()
    ref_cpu = encode_taxels(
        contact_pos=torch.as_tensor(p_pos),
        contact_frame=torch.as_tensor(p_frame),
        contact_force=torch.as_tensor(p_force),
        contact_sign=torch.as_tensor(p_sign),
        contact_body=torch.as_tensor(p_body),
        contact_valid=torch.as_tensor(p_valid),
        site_pos=torch.as_tensor(site_pos),
        site_mat=torch.as_tensor(site_mat),
        taxel_body=torch.as_tensor(enc.static.taxel_body),
        kernel_sigma=KERNEL_SIGMA,
        kernel_cutoff=KERNEL_CUTOFF,
    ).numpy()
    err_a = float(np.abs(got - ref_cpu).max())

    # ---- B: the supervisor's own encoder on world 0 ------------------------- #
    from leapxela.taxel_layout import build_layout
    from leapxela.touch_sensor import VirtualTaxelSensor
    from taxel_map import remap_layout

    mjd = mj.MjData(mjm)
    mjd.qpos[:] = d.qpos.numpy()[0].astype(np.float64)
    mjd.qvel[:] = d.qvel.numpy()[0].astype(np.float64)
    mjd.ctrl[:] = d.ctrl.numpy()[0].astype(np.float64)
    mj.mj_forward(mjm, mjd)
    sensor = VirtualTaxelSensor(
        mjm, remap_layout(build_layout()), cube_geom_names(mjm), KERNEL_SIGMA, KERNEL_CUTOFF
    )
    ref_sensor = np.asarray(sensor.update(mjd), dtype=np.float64)
    err_b = float(np.abs(got[0] - ref_sensor).max())

    gpu_pairs = sorted(map(tuple, c_geom[c_world == 0]))
    cpu_pairs = sorted((int(mjd.contact[i].geom1), int(mjd.contact[i].geom2))
                       for i in range(mjd.ncon))

    return dict(
        check_a_abs_err=err_a,
        check_a_pass=bool(err_a < tol),
        check_b_abs_err=err_b,
        check_b_gpu_ncon_world0=int((c_world == 0).sum()),
        check_b_cpu_ncon=int(mjd.ncon),
        check_b_same_geom_pairs=bool(gpu_pairs == cpu_pairs),
        gpu_active_taxels_world0=int((np.linalg.norm(got[0], axis=1) > 1e-6).sum()),
        cpu_active_taxels=int((np.linalg.norm(ref_sensor, axis=1) > 1e-6).sum()),
        gpu_peak_taxel_n=float(np.linalg.norm(got, axis=-1).max()),
        cpu_peak_taxel_n=float(np.linalg.norm(ref_sensor, axis=1).max()),
        tol=tol,
    )


def verify_touch(mjm, d, touch: TouchReadout) -> dict:
    """Compare Warp's touch sensors against MuJoCo's own, on world 0.

    This is the touch analogue of `verify`'s check B, and it carries the same
    caveat: REPORTED, NOT GATED. Copying world 0's state into an `MjData` and
    calling `mj_forward` makes the CPU re-run collision detection and re-solve
    the constraints in float64, so it finds its own contact set and its own
    `efc_force`. A touch sensor is a sum of normal forces over contacts inside a
    1 mm sphere, so any disagreement in either propagates straight through.
    What this check can catch -- and the only thing it is asked to catch -- is a
    wiring mistake big enough to make the two answers unrelated.

    There is no check-A analogue here. For splat, check A re-implements the
    encoder on the host and so tests OUR code. Here the sensor is
    mujoco_warp's; we contribute the model and the readout, and nothing else.
    """
    active_gate = 0.0
    got = touch.read().detach().cpu().numpy()[0]

    mjd = mj.MjData(mjm)
    mjd.qpos[:] = d.qpos.numpy()[0].astype(np.float64)
    mjd.qvel[:] = d.qvel.numpy()[0].astype(np.float64)
    mjd.ctrl[:] = d.ctrl.numpy()[0].astype(np.float64)
    mj.mj_forward(mjm, mjd)
    ref = mjd.sensordata[touch.adr.cpu().numpy()]

    return dict(
        touch_check_abs_err=float(np.abs(got - ref).max()),
        touch_check_rel_err=float(
            np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12)),
        touch_check_gpu_active=int((got > active_gate).sum()),
        touch_check_cpu_active=int((ref > active_gate).sum()),
        touch_check_gpu_total_n=float(got.sum()),
        touch_check_cpu_total_n=float(ref.sum()),
        touch_check_gpu_ncon=int((d.contact.worldid.numpy()[:int(d.nacon.numpy()[0])] == 0).sum()),
        touch_check_cpu_ncon=int(mjd.ncon),
    )


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--scene", type=pathlib.Path, default=DEFAULT_SCENE)
    ap.add_argument("--encoder", choices=["torch", "compile", "triton"],
                    default="torch",
                    help="splat backend; all three are numerically equivalent "
                         "(see tests/test_triton_kernel.py)")
    ap.add_argument("--tactile", choices=["none", "splat", "touch", "touchgrid"],
                    default="none",
                    help="representation under test: 'none' is the H1 control")
    ap.add_argument("--touch-scene", choices=["inject", "supervisor"], default="inject",
                    help="only for --tactile touch. 'inject' puts the 421 touch "
                         "sites into the RL scene the other sweeps use; "
                         "'supervisor' uses the standalone touch model, whose "
                         "sensors read zero (see build_touch_model)")
    ap.add_argument("--touchgrid-collide", choices=["object", "naive", "off"],
                    default="object",
                    help="only for --tactile touchgrid. 'object' = patches sense "
                         "the cube only (headline); 'naive' = the donor's own "
                         "1/1 masks, which also self-collide; 'off' = the "
                         "non-colliding model-size control. See "
                         "build_touchgrid_model")
    ap.add_argument("--nconmax-per-env", type=int, default=NCONMAX_PER_ENV,
                    help="contact budget per world. Defaults to the supervisor's "
                         "48; raise it only to measure what a representation "
                         "actually needs, and say so in the write-up")
    ap.add_argument("--njmax-per-env", type=int, default=NJMAX_PER_ENV,
                    help="constraint budget per world; the supervisor's is 120")
    ap.add_argument("--motion", choices=["static", "grasp"], default="static",
                    help="'static' is the frozen 60%%-closure pose every banked "
                         "sweep used; 'grasp' drives the supervisor's seven "
                         "grasp patterns with worlds decorrelated (GraspDriver)")
    ap.add_argument("--allow-solver-drift", action="store_true",
                    help="run even if solver settings differ from the "
                         "supervisor's env_cfg.py (they are then reported, and "
                         "the numbers are not comparable to his training)")
    ap.add_argument("--verify", action="store_true",
                    help="run the CPU-reference correctness gate before timing")
    args = ap.parse_args()

    t_import = time.perf_counter()
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING   # module-load chatter would drown the JSON
    import mujoco_warp as mjw
    import torch
    import_s = time.perf_counter() - t_import

    n = args.num_envs
    tactile = args.tactile
    print(f"[harness] N={n} scene={args.scene.name} tactile={tactile}", file=sys.stderr)

    tg_collide = args.touchgrid_collide
    nconmax_pe, njmax_pe = args.nconmax_per_env, args.njmax_per_env
    mjm = build_scene_model(args.scene, tactile, args.touch_scene, tg_collide)
    mjm.opt.timestep = SIM_DT
    solver_info = assert_solver_matches_supervisor(
        mjm, strict=not args.allow_solver_drift)
    mjd = mj.MjData(mjm)
    mj.mj_forward(mjm, mjd)

    t0 = time.perf_counter()
    m = mjw.put_model(mjm)
    d = mjw.put_data(
        mjm, mjd, nworld=n,
        naconmax=nconmax_pe * n,        # total across worlds
        njmax=njmax_pe,                 # per world -- do not scale
    )
    wp.synchronize()
    setup_s = time.perf_counter() - t0
    rss_after_alloc = peak_rss_gb()

    enc = touch = tgrid = None
    if tactile == "touchgrid":
        # Same stream argument as the other two paths: the readout is a torch
        # scatter over the Warp contact arrays, with no sync in the fused loop.
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()), wp.get_device())
        tgrid = TouchgridReadout(mjm, m, d, n)
        torch.cuda.reset_peak_memory_stats()
    if tactile == "touch":
        # Same reason as the splat path: the readout is a torch op reading a
        # Warp allocation, so Warp must be on torch's stream or the fused loop
        # can read sensordata before `_sensor_touch` has finished writing it.
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()), wp.get_device())
        touch = TouchReadout(mjm, m, d, n, mjw)
        torch.cuda.reset_peak_memory_stats()
    if tactile == "splat":
        # Warp and torch default to different CUDA streams. The fused timing
        # loop has no sync between `mjw.step` and the torch encoder, so without
        # this the encoder could read contact arrays the physics kernels have
        # not finished writing. Put Warp on torch's stream and the ordering is
        # the stream's problem, not ours.
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()), wp.get_device())
        # `contacts_per_env` must follow the budget this run actually allocated,
        # not the module constant. Without it, `--nconmax-per-env 256` would
        # allocate 256 slots per world in `put_data` above while the encoder
        # bucketed into 48 -- silently discarding every contact past the 48th and
        # reporting it as a drop rather than as a misconfiguration. Harmless for
        # the banked sweeps, which all ran at the default 48, which is exactly
        # why it survived unnoticed.
        enc = _attach_mjw(
            SplatEncoder(mjm, m, d, n, encoder=args.encoder,
                         contacts_per_env=nconmax_pe),
            mjw,
        )
        torch.cuda.reset_peak_memory_stats()

    # ---- close the hand so contacts exist ----------------------------------
    driver = None
    if args.motion == "grasp":
        driver = GraspDriver(mjm, d, n, SIM_DT)
        for k in range(driver.n_settle):
            driver.settle_step(k)
            mjw.step(m, d)
    else:
        lo = mjm.actuator_ctrlrange[:, 0]
        hi = mjm.actuator_ctrlrange[:, 1]
        target = lo + CLOSE_FRACTION * (hi - lo)
        d.ctrl.assign(np.tile(target.astype(np.float32), (n, 1)))
        for _ in range(SETTLE_STEPS):
            mjw.step(m, d)
    wp.synchronize()
    contacts_now = int(d.nacon.numpy()[0])

    # ---- correctness gate: no timing is worth anything if the answer is wrong
    checks = None
    touch_stats = None
    tgrid_stats = None
    if tgrid is not None:
        # Unconditional for the same reason as the touch gate. The failure mode
        # here is different but just as silent: with `--touchgrid-collide off`
        # the patches cannot register anything, and that variant is *supposed*
        # to read zero -- it is the model-size control, not a sensor. So only
        # the colliding variant is gated.
        tgrid.read()
        tgrid_stats = tgrid.signal_stats()
        tgrid_stats.update(tgrid.contact_mix(mjm))
        # The `object` mask lets a patch reach the floor on paper. Check rather
        # than argue: if this is ever non-zero the mask reasoning in
        # `build_touchgrid_model` is wrong and the numbers are not what they claim.
        if tgrid_stats["touchgrid_patch_floor_contacts"]:
            print("[harness] WARNING: "
                  f"{tgrid_stats['touchgrid_patch_floor_contacts']} patch-floor "
                  "contacts -- the collision mask argument does not hold here",
                  file=sys.stderr)
        print(f"[harness] touchgrid: {tgrid_stats['touchgrid_active_world0']}/"
              f"{tgrid_stats['n_patches']} patches in contact in world 0, "
              f"{tgrid_stats['touchgrid_active_any_world']} in at least one world, "
              f"{tgrid_stats['touchgrid_active_mean_per_world']:.2f} mean/world",
              file=sys.stderr)
        if tg_collide != "off" and tgrid_stats["touchgrid_active_any_world"] == 0:
            print(json.dumps(dict(num_envs=n, tactile=tactile, ok=False,
                                  touchgrid_collide=args.touchgrid_collide,
                                  note="no touchgrid patch ever in contact",
                                  contacts=contacts_now, **tgrid_stats)))
            print(f"[harness] STOPPING: none of the {tgrid_stats['n_patches']} "
                  f"patches registers a contact at a gripping pose "
                  f"({contacts_now} live contacts).", file=sys.stderr)
            return 2

    if touch is not None:
        # Unconditional, not behind --verify: a touch array that reads zero
        # benchmarks identically to one that works, and the supervisor's own
        # standalone touch model does exactly that (see `build_touch_model`).
        # Nothing downstream is worth measuring if this fails.
        touch.read()
        touch_stats = touch.signal_stats()
        if driver is not None and touch_stats["touch_active_any_world"] == 0:
            # A single instant is the right check for `--motion static`, where
            # the grasp is held and contact is guaranteed. It is the WRONG check
            # under `--motion grasp`: the patterns deliberately release (`tap`
            # drops 50% back toward pregrip at 1.5 Hz), so a quiet instant means
            # the hand is momentarily open, not that the sensors are broken.
            # That false alarm aborted the first grasp-motion touch sweep at
            # N=1, where there is one world running one pattern at one phase.
            # Sample a window and take the best frame; the gate still catches a
            # sensor array that never fires, which is what it is for.
            probe = min(driver.n_cycle, 200)
            best = touch_stats
            for _ in range(probe):
                driver.step()
                mjw.step(m, d)
                touch.read()
                s = touch.signal_stats()
                if s["touch_active_any_world"] > best["touch_active_any_world"]:
                    best = s
                if best["touch_active_any_world"]:
                    break
            touch_stats = best
            driver.reset_phase()
        print(f"[harness] touch: {touch_stats['touch_active_world0']}/"
              f"{touch_stats['n_touch_sensors']} sensors non-zero in world 0, "
              f"{touch_stats['touch_active_any_world']} non-zero in at least one world, "
              f"{touch_stats['touch_active_mean_per_world']:.2f} mean/world, "
              f"range {touch_stats['touch_min_nonzero_n']:.4f}-"
              f"{touch_stats['touch_max_n']:.4f} N", file=sys.stderr)
        if touch_stats["touch_active_any_world"] == 0:
            print(json.dumps(dict(num_envs=n, tactile=tactile, ok=False,
                                  touch_scene=args.touch_scene,
                                  note="all touch sensors read zero", contacts=contacts_now,
                                  **touch_stats)))
            print("[harness] STOPPING: every one of the "
                  f"{touch_stats['n_touch_sensors']} touch sensors reads zero at a "
                  f"gripping pose ({contacts_now} live contacts). Timing a sensor "
                  "that never fires measures nothing.", file=sys.stderr)
            return 2

    if args.verify:
        if tgrid is not None:
            checks = verify_touchgrid(mjm, d, tgrid)
            print(f"[harness] verify: touchgrid GPU vs CPU mj_forward on world 0, "
                  f"active gpu/cpu {checks['touchgrid_check_gpu_active']}/"
                  f"{checks['touchgrid_check_cpu_active']}  "
                  f"patches disagreeing {checks['touchgrid_check_disagree']}/"
                  f"{tgrid.S}  ncon gpu/cpu {checks['touchgrid_check_gpu_ncon']}/"
                  f"{checks['touchgrid_check_cpu_ncon']}  (reported, not gated)",
                  file=sys.stderr)
        elif touch is not None:
            checks = verify_touch(mjm, d, touch)
            print(f"[harness] verify: touch GPU vs CPU mj_forward on world 0, "
                  f"max abs err {checks['touch_check_abs_err']:.3e}  "
                  f"active gpu/cpu {checks['touch_check_gpu_active']}/"
                  f"{checks['touch_check_cpu_active']}  "
                  f"ncon gpu/cpu {checks['touch_check_gpu_ncon']}/"
                  f"{checks['touch_check_cpu_ncon']}  (reported, not gated)",
                  file=sys.stderr)
        elif enc is None:
            print("[harness] --verify only applies to the tactile paths", file=sys.stderr)
        else:
            checks = verify(mjm, m, d, enc)
            print(f"[harness] verify: check A (GPU vs CPU encode_taxels, same contacts) "
                  f"max abs err {checks['check_a_abs_err']:.3e} "
                  f"{'PASS' if checks['check_a_pass'] else 'FAIL'}", file=sys.stderr)
            print(f"[harness] verify: check B (vs VirtualTaxelSensor, CPU re-solve) "
                  f"max abs err {checks['check_b_abs_err']:.3e}  "
                  f"ncon gpu/cpu {checks['check_b_gpu_ncon_world0']}/{checks['check_b_cpu_ncon']}"
                  f"  same geom pairs: {checks['check_b_same_geom_pairs']}  (reported, not gated)",
                  file=sys.stderr)
            if not checks["check_a_pass"]:
                print(json.dumps(dict(num_envs=n, tactile=tactile, ok=False,
                                      note="verify FAILED", **checks)))
                print("[harness] STOPPING: GPU tactile path disagrees with the CPU "
                      "reference. A fast wrong number is worth nothing.", file=sys.stderr)
                return 2

    # ---- warm up: first steps pay for kernel codegen, not physics ----------
    def _sync():
        wp.synchronize()
        torch.cuda.synchronize()

    # Whichever readout this run has, if any. Both the touch and touchgrid
    # splits are differences of fused loops, so they need the same helper.
    _readout = touch.read if touch is not None else (
        tgrid.read if tgrid is not None else None)

    def _timed(steps: int, step: bool = True, readout: bool = False) -> float:
        """Wall-clock ms/step for a fused loop. Syncs only at the boundaries."""
        if driver is not None:
            driver.reset_phase()
        _sync()
        t = time.perf_counter()
        for _ in range(steps):
            if driver is not None and step:
                driver.step()
            if step:
                mjw.step(m, d)
            if readout:
                _readout()
        _sync()
        return 1000.0 * (time.perf_counter() - t) / steps

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        if driver is not None:
            driver.step()
        mjw.step(m, d)
        if enc is not None:
            enc.encode()
        if _readout is not None:
            _readout()
    if touch is not None:
        # Each switch position is a different set of kernel launches, so warm
        # all of them -- otherwise the first loop of a configuration pays for
        # module load and codegen and shows up as that configuration's cost.
        for touch_on, sensors_on in ((True, True), (False, True), (True, False)):
            touch.set_touch(touch_on)
            touch.set_sensors(sensors_on)
            for _ in range(max(4, args.warmup // 4)):
                mjw.step(m, d)
        touch.set_touch(True)
        touch.set_sensors(True)
    wp.synchronize()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0

    contacts = int(d.nacon.numpy()[0])
    overflow = int((d.overflow.numpy() != 0).sum())

    # ---- contact statistics over the motion, sampled OUTSIDE the clock ------
    # With `--motion grasp` the contact count is no longer a constant, so the
    # single snapshot above stops describing the run: it is whatever the last
    # step happened to leave behind. This replays the same motion with a host
    # read per step -- far too expensive to sit inside a timed loop, which is
    # exactly why it lives here instead.
    contact_stats = {}
    if driver is not None:
        driver.reset_phase()
        seen, quietest, busiest = [], [], []
        for _ in range(min(args.steps, 200)):
            driver.step()
            mjw.step(m, d)
            wp.synchronize()
            nac = int(d.nacon.numpy()[0])
            seen.append(nac)
            # Per-world spread at this instant. This, not the total, is what the
            # frozen pose could never produce and what a gather kernel actually
            # feels: one block per (env, taxel tile), so an env with 26 contacts
            # and an env with 0 in the same launch is a straggler problem.
            per = np.bincount(d.contact.worldid.numpy()[:nac], minlength=n)
            quietest.append(int(per.min()))
            busiest.append(int(per.max()))
        seen = np.asarray(seen, dtype=np.float64)
        contact_stats = dict(
            contacts_min=int(seen.min()),
            contacts_max=int(seen.max()),
            contacts_mean=float(seen.mean()),
            contacts_per_env_min=float(seen.min() / n),
            contacts_per_env_max=float(seen.max() / n),
            contacts_per_env_mean=float(seen.mean() / n),
            world_contacts_min=int(np.min(quietest)),
            world_contacts_max=int(np.max(busiest)),
            contact_sample_steps=int(seen.size),
        )
        overflow = int((d.overflow.numpy() != 0).sum())

    # ---- timed, pass 1: fused, no inner syncs -> headline throughput --------
    wp.synchronize()
    torch.cuda.synchronize()
    if driver is not None:
        driver.reset_phase()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        if driver is not None:
            driver.step()
        mjw.step(m, d)
        if enc is not None:
            enc.encode()
        if _readout is not None:
            _readout()
    wp.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # ---- timed, touchgrid: step-only and readout-only, both fused -----------
    # Simpler than the touch split because there is nothing to gate: MuJoCo runs
    # no sensor stage for a touchgrid, so the readout really is a separate torch
    # op after `mjw.step` and a fused step-only loop is a clean baseline. The
    # cost H3 is about does not appear in this split at all -- it is inside
    # `step_only_ms`, and it is visible only by comparing against the
    # `--touchgrid-collide off` run and the banked physics-only sweep.
    tgrid_split = {}
    if tgrid is not None:
        step_ms = _timed(args.steps)
        step_ms_b = _timed(args.steps)                    # repeat -> noise floor
        read_ms = _timed(args.steps, step=False, readout=True)
        tgrid_split = dict(
            touchgrid_collide=args.touchgrid_collide,
            step_only_ms_per_step=step_ms,
            readout_ms_per_step=read_ms,
            timing_noise_ms_per_step=abs(step_ms - step_ms_b),
        )

    # ---- timed, touch: the same loop with pieces switched off ---------------
    # See the module docstring. Warp computes touch sensors inside `mjw.step`,
    # so the split is a difference of fused loops, not a stopwatch around a
    # stage. All four loops below run identical dynamics -- sensor output never
    # feeds back into the state -- so they are directly subtractable.
    touch_split = {}
    if touch is not None:
        head_ms = 1000.0 * elapsed / args.steps          # step + readout, all on
        full_ms = _timed(args.steps)                     # step only, all on
        touch.set_touch(False)
        notouch_ms = _timed(args.steps)                  # _sensor_touch launched with size 0
        touch.set_touch(True)
        touch.set_sensors(False)
        nosens_ms = _timed(args.steps)                   # whole sensor stage skipped
        nosens_ms_b = _timed(args.steps)                 # repeat -> noise floor
        touch.set_sensors(True)
        read_ms = _timed(args.steps, step=False, readout=True)

        touch_split = dict(
            head_ms_per_step=head_ms,
            step_all_on_ms_per_step=full_ms,
            step_touch_off_ms_per_step=notouch_ms,
            step_sensors_off_ms_per_step=nosens_ms,
            touch_kernel_ms_per_step=full_ms - notouch_ms,
            sensor_stage_ms_per_step=notouch_ms - nosens_ms,
            readout_ms_per_step=read_ms,
            timing_noise_ms_per_step=abs(nosens_ms - nosens_ms_b),
        )

    # ---- timed, pass 2: synchronised between stages -> the split ------------
    # The inner syncs drain the pipeline twice per step, so `split_ms_per_step`
    # is an upper bound on the fused cost. Reported so the gap is visible rather
    # than quietly folded into one of the two halves.
    if tgrid is not None:
        # Baseline = the same loop without the readout. Note what this means and
        # what it does not: `physics_ms` here is the step time *including* the
        # 300 patch geoms, so it is NOT comparable to the physics-only sweep's
        # `physics_ms` -- that difference is the H3 finding, and reporting it as
        # part of the encoder would hide it in the one place it must be visible.
        physics_s = args.steps * tgrid_split["step_only_ms_per_step"] / 1000.0
        encoder_s = elapsed - physics_s
    elif touch is not None:
        # Baseline = the identical loop with only the 421 touch sensors gated
        # out. Same scene, same contacts, same 29 pose sensors the physics-only
        # sweep already ran, so `encoder_ms` is the marginal cost of the touch
        # array and nothing else. By construction physics + encoder == the
        # headline total, exactly.
        physics_s = args.steps * touch_split["step_touch_off_ms_per_step"] / 1000.0
        encoder_s = elapsed - physics_s
    elif enc is None:
        physics_s, encoder_s = elapsed, 0.0
    else:
        physics_s = encoder_s = 0.0
        if driver is not None:
            driver.reset_phase()
        for _ in range(args.steps):
            if driver is not None:
                driver.step()
            t = time.perf_counter()
            mjw.step(m, d)
            wp.synchronize()
            torch.cuda.synchronize()
            physics_s += time.perf_counter() - t

            t = time.perf_counter()
            enc.encode()
            torch.cuda.synchronize()
            wp.synchronize()
            encoder_s += time.perf_counter() - t

    env_steps = n * args.steps
    result = dict(
        num_envs=n,
        tactile=tactile,
        steps=args.steps,
        warmup=args.warmup,
        scene=args.scene.name,
        sim_dt=float(mjm.opt.timestep),
        nconmax_per_env=nconmax_pe,
        njmax_per_env=njmax_pe,
        naconmax_total=int(d.naconmax),
        njmax_actual=int(d.njmax),
        contacts=contacts,
        contacts_per_env=contacts / n,
        overflow_worlds=overflow,
        wall_s=elapsed,
        ms_per_step=1000.0 * elapsed / args.steps,
        env_steps_per_sec=env_steps / elapsed,
        us_per_env_step=1e6 * elapsed / env_steps,
        physics_ms_per_step=1000.0 * physics_s / args.steps,
        encoder_ms_per_step=1000.0 * encoder_s / args.steps,
        encoder_frac=encoder_s / (physics_s + encoder_s) if (physics_s + encoder_s) else 0.0,
        split_ms_per_step=1000.0 * (physics_s + encoder_s) / args.steps,
        setup_s=setup_s,
        warmup_s=warmup_s,
        import_s=import_s,
        peak_rss_gb=peak_rss_gb(),
        rss_after_alloc_gb=rss_after_alloc,
        torch_peak_gb=torch.cuda.max_memory_allocated() / 1024**3,
        device=str(wp.get_device()),
        motion=args.motion,
        **solver_info,
        ok=True,
        **contact_stats,
    )
    if tgrid is not None:
        result.update(ngeom=int(mjm.ngeom), **tgrid_split, **tgrid_stats)
    if touch is not None:
        result.update(touch_scene=args.touch_scene, **touch_split, **touch_stats)
    if enc is not None:
        active, peak = enc.signal_stats()
        result.update(
            n_taxels=enc.T,
            contacts_per_env_cap=enc.C,
            dropped_contacts=enc.dropped_contacts(),
            active_taxels_world0=active,
            peak_taxel_n=peak,
        )
    if checks:
        result.update(checks)

    print(json.dumps(result))
    print(f"[harness] {result['env_steps_per_sec']:,.0f} env-steps/s  "
          f"{result['ms_per_step']:.2f} ms/step  peak {result['peak_rss_gb']:.2f} GB  "
          f"contacts {contacts} ({contacts / n:.1f}/env)  overflow {overflow}",
          file=sys.stderr)
    if tgrid is not None:
        print(f"[harness]   step({result['ngeom']} geoms, collide="
              f"{result['touchgrid_collide']}) {result['step_only_ms_per_step']:.3f} ms  "
              f"readout {result['readout_ms_per_step']:.3f} ms "
              f"({100 * result['encoder_frac']:.1f}%)  "
              f"noise floor +-{result['timing_noise_ms_per_step']:.3f} ms  "
              f"patches in contact {result['touchgrid_active_world0']}/"
              f"{result['n_patches']} (w0)  overflow {overflow}", file=sys.stderr)
    if touch is not None:
        print(f"[harness]   baseline(touch off) {result['step_touch_off_ms_per_step']:.3f} ms  "
              f"touch kernel {result['touch_kernel_ms_per_step']:.3f} ms  "
              f"readout {result['readout_ms_per_step']:.3f} ms  "
              f"[mjwarp sensor stage {result['sensor_stage_ms_per_step']:.3f} ms, "
              f"charged to baseline]  "
              f"total tactile {result['encoder_ms_per_step']:.3f} ms "
              f"({100 * result['encoder_frac']:.1f}%)  "
              f"noise floor +-{result['timing_noise_ms_per_step']:.3f} ms  "
              f"active {result['touch_active_world0']}/{result['n_touch_sensors']} (w0)  "
              f"peak {result['touch_max_n']:.3f} N", file=sys.stderr)
    if enc is not None:
        print(f"[harness]   physics {result['physics_ms_per_step']:.2f} ms  "
              f"encoder {result['encoder_ms_per_step']:.2f} ms  "
              f"encoder share {100 * result['encoder_frac']:.1f}%  "
              f"torch peak {result['torch_peak_gb']:.2f} GB  "
              f"active taxels(w0) {result['active_taxels_world0']}  "
              f"peak {result['peak_taxel_n']:.3f} N  "
              f"dropped {result['dropped_contacts']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
