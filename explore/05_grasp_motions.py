"""Purpose-built grasp motions, and the extended oracle fixture they produce.

WHY: `04_run_encoder.py` banks the fixture that every encoder implementation is
validated against, but it generates it with a crude scripted motion -- drive all
16 actuators to 60% of range, let the cube settle. Its coverage is poor:

    37 of 368 taxels ever fire (31 palm, 4 proximal, 2 fingertip)
    8 of 30 frames have ncon == 0, max 14 contacts against a 48-slot budget
    almost pure normal force; the two shear channels are barely excited

That matters most for the 136 FINGERTIP taxels. The fingertip bodies are the
`SUSPECT` set in `taxel_map.py`: the tactile and RL lineages disagree there (the
RL tip is a physically different, ~10.3 mm longer part), so fingertip placement
is the least certain part of the whole system -- and it is exactly what the old
fixture fails to exercise. A bug confined to the fingertip path passes every
current test.

WHAT THIS IS: a port of the seven grasp patterns the supervisor wrote for
tactile data collection, in

    https://github.com/mohammad200h/sparsh-skin-sim -> util/motion_util.py

(`GRASP_PATTERNS`, `GraspProfile`, `grasp_target`, `finger_close_grasp_target`).
His code targets the *flex* model in that repo, behind an `ActuatorModel`
protocol; this drives the pinned rigid hand + cube scene that `04_run_encoder.py`
uses. The pattern shapes, timings and fractions are HIS -- every constant below
is copied from his module rather than re-tuned, because the point is to use the
motions he designed, not to invent lookalikes with different numbers. What is
ported is the plumbing: `mj.MjModel` instead of the protocol, and the scene
reset/settle around each pattern.

The two patterns that earn their place here:
  * `shear` -- modulates the abduction/axial-roll actuators, which drags the
    contact patch tangentially across the taxels and is the only thing in the
    set that genuinely excites shear_x / shear_y;
  * `tap` -- releases 50% of the way back to pregrip at 1.5 Hz, so contact
    counts sweep up and down through zero instead of sitting at a fixed grasp.

Writes `explore/out/oracle_fixture_v2.npz`, same schema as the original (it
reuses `04_run_encoder.py`'s own `capture`/`save_fixture`, so the two files
cannot drift apart). The original is NOT touched.

WHAT IT ACHIEVED (measured, not hoped for — `main` reprints this every run):

                        v1 (04_run_encoder)     v2 (this script)
    frames                    30                     632
    taxels ever live          37 / 368               176 / 368
      fingertips               2 / 136                57 / 136
      thumb (th_px+th_ds)      0 / 62                 11 / 62
    contacts/frame       0 / 2.5 / 14           0 / 7 / 19  (min/med/max)
    ncon == 0 frames           8                      16
    shear_x range        -1.08 .. +0.94         -5.04 .. +2.15
    shear_y range        -2.46 .. +0.69         -5.42 .. +4.09

WHAT IT REVEALED — three things, none of which the old fixture could see:

 1. THE MEDIAL PHALANGES ARE UNREACHABLE. Every cube contact on `if_md`/`mf_md`
    is exactly 11.50 mm out of the taxel plane, against a `kernel_cutoff` of
    10 mm — a constant, so 0 of 32 medial contacts ever reach a taxel and those
    48 taxels can never fire in this scene no matter what motion is run.
 2. THE FINGERTIPS SIT ON THE CUTOFF. Fingertip contacts land ~9-10.8 mm out of
    plane from the nearest taxel: right at the gate. 33-49% of fingertip
    contacts produce nothing at all. This is the `taxel_map.SUSPECT` / D4
    geometry mismatch (RL tip ~10.3 mm longer) showing up quantitatively: the
    cube touches the extra distal length, which carries no taxels.
 3. SO DOES THE PALM, AND THAT BREAKS float32 AGREEMENT. Palm taxels sit at
    |local_z| = 10.0000 mm — margin to the gate measured at 0.0-0.2 um. float32
    puts a taxel on the other side of the hard `|local_z| > cutoff` gate from
    the float64 reference, and because the surviving weights are renormalised,
    gaining or losing one taxel from a 2-5 taxel splat rescales the rest by up
    to 4.6%. See `tests/test_encoder_oracle.py` for the full write-up and the
    float64 control that proves it is precision, not logic.

Run (memory-capped, as everything on this box must be -- unified memory means an
oversized allocation takes down the host, not the process):

    PYTHONPATH=third_party/leapXELA_model:explore \
      systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
      .venv/bin/python explore/05_grasp_motions.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import importlib.util
import pathlib
from dataclasses import dataclass
from typing import Iterable

import mujoco as mj
import numpy as np

from leapxela.taxel_layout import build_layout
from leapxela.touch_sensor import VirtualTaxelSensor
from taxel_map import BODY_MAP, SUSPECT, remap_layout

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "out"


def _load_run_encoder():
    """Import `04_run_encoder.py` by path — its name is not an identifier.

    Reusing its `build_model`, `cube_geom_names`, `capture` and `save_fixture`
    rather than copying them is deliberate: the two fixtures must be produced by
    the same model and written in the same schema, or the v2 file silently stops
    being comparable to v1.
    """
    path = ROOT / "explore" / "04_run_encoder.py"
    spec = importlib.util.spec_from_file_location("run_encoder_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _load_run_encoder()
KERNEL_SIGMA = REF.KERNEL_SIGMA
KERNEL_CUTOFF = REF.KERNEL_CUTOFF


# --------------------------------------------------------------------------- #
# Ported from mohammad200h/sparsh-skin-sim util/motion_util.py
# --------------------------------------------------------------------------- #

GRASP_PATTERNS = (
    "hold",
    "pulse",
    "squeeze",
    "regrasp",
    "tap",
    "shear",
    "finger_close",
)

# Index / middle / ring flexion only — no rot or thumb joints. Our RL model
# happens to use the same joint names as his flex model, so this transfers
# verbatim; `_actuator_id` below still checks rather than assuming.
FINGER_CLOSE_JOINTS: tuple[str, ...] = (
    "if_mcp", "if_pip", "if_dip",
    "mf_mcp", "mf_pip", "mf_dip",
    "rf_mcp", "rf_pip", "rf_dip",
)

DEFAULT_CLOSE_DURATION = 1.0
DEFAULT_MCP_TARGET = 1.6
DEFAULT_PIP_TARGET = 1.5
DEFAULT_DIP_TARGET = 0.9
FINGER_CLOSE_TH_AXL = 1.6

CLOSE_START = 0.0
CLOSE_END = 1.5
PREGRIP_FRACTION = 0.40
GRIP_FRACTION = 0.85
THUMB_GRIP_FRACTION = 0.35
THUMB_DELAY = 0.5
PULSE_HZ = 1.0
PULSE_AMPLITUDE = 0.15
SHEAR_AMPLITUDE = 0.15
SQUEEZE_STEPS = 4
SQUEEZE_STEP_SECONDS = 0.9
REGRASP_HZ = 0.35
TAP_HZ = 1.5
TAP_DEPTH = 0.5
SHEAR_HZ = 0.7


@dataclass(frozen=True)
class GraspProfile:
    """Parameters controlling one grasp pattern (his dataclass, same fields)."""

    pattern: str
    grip_fraction: float = GRIP_FRACTION
    thumb_grip_fraction: float = THUMB_GRIP_FRACTION
    pregrip_fraction: float = PREGRIP_FRACTION
    thumb_delay: float = THUMB_DELAY
    pulse_hz: float = PULSE_HZ
    pulse_amplitude: float = PULSE_AMPLITUDE
    shear_amplitude: float = SHEAR_AMPLITUDE
    squeeze_steps: int = SQUEEZE_STEPS
    close_duration: float = DEFAULT_CLOSE_DURATION
    mcp_target: float = DEFAULT_MCP_TARGET
    pip_target: float = DEFAULT_PIP_TARGET
    dip_target: float = DEFAULT_DIP_TARGET
    hold_after_close: bool = True


def validate_profile(profile: GraspProfile) -> None:
    if profile.pattern not in GRASP_PATTERNS:
        raise ValueError(
            f"Unknown grasp pattern '{profile.pattern}'. Expected one of {GRASP_PATTERNS}"
        )
    for name, value in (
        ("grip_fraction", profile.grip_fraction),
        ("thumb_grip_fraction", profile.thumb_grip_fraction),
        ("pregrip_fraction", profile.pregrip_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if profile.squeeze_steps < 1:
        raise ValueError("squeeze_steps must be at least 1")
    if profile.close_duration <= 0.0:
        raise ValueError("close_duration must be positive")


def _actuator_id(model: mj.MjModel, joint_name: str) -> int:
    act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_act")
    if act_id < 0:
        raise ValueError(f"Actuator '{joint_name}_act' not found")
    return int(act_id)


def resolve_actuator_ids(model: mj.MjModel, joint_names: Iterable[str]) -> np.ndarray:
    return np.asarray([_actuator_id(model, n) for n in joint_names], dtype=np.int32)


def actuator_names(model: mj.MjModel) -> list[str]:
    return [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)
    ]


def thumb_mask(model: mj.MjModel) -> np.ndarray:
    """Thumb actuators, by the LeapXELA naming convention (`th_*`)."""
    return np.asarray([n.startswith("th_") for n in actuator_names(model)], dtype=bool)


def lateral_mask(model: mj.MjModel) -> np.ndarray:
    """Abduction / axial-roll actuators — the ones `shear` modulates.

    In our model these are `if_rot_act`, `mf_rot_act`, `rf_rot_act`, `th_axl_act`:
    the four joints that move a fingertip *sideways* across the cube face rather
    than into it, which is what puts force in the tangential channels.
    """
    names = actuator_names(model)
    return np.asarray(["_rot_" in n or "_axl_" in n for n in names], dtype=bool)


def pregrip_targets(model: mj.MjModel, pregrip_fraction: float) -> np.ndarray:
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return low + pregrip_fraction * (high - low)


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _finger_close_closed_targets(names, lo, hi, *, mcp_target, pip_target, dip_target):
    targets = np.empty(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        if name.endswith("_mcp"):
            targets[i] = mcp_target
        elif name.endswith("_pip"):
            targets[i] = pip_target
        elif name.endswith("_dip"):
            targets[i] = dip_target
        else:
            raise ValueError(f"finger_close only supports mcp/pip/dip joints, got '{name}'")
    return np.clip(targets, lo, hi)


def finger_close_grasp_target(
    model: mj.MjModel,
    time_seconds: float,
    profile: GraspProfile,
    *,
    joint_names: Iterable[str] = FINGER_CLOSE_JOINTS,
    start_ctrl: np.ndarray | None = None,
) -> np.ndarray:
    """Full actuator target vector for the `finger_close` pattern.

    Only IF/MF/RF flexion is driven; the thumb sits at `FINGER_CLOSE_TH_AXL` on
    its axial joint and everything else at zero. This is the one pattern that
    presses fingers into the cube without a thumb opposing them.
    """
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    if start_ctrl is None:
        targets = np.zeros(model.nu, dtype=np.float64)
        th_axl_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "th_axl_act")
        if th_axl_id >= 0:
            targets[th_axl_id] = np.clip(
                FINGER_CLOSE_TH_AXL, low[th_axl_id], high[th_axl_id]
            )
    else:
        targets = np.asarray(start_ctrl, dtype=np.float64).copy()

    names = tuple(joint_names)
    act_ids = resolve_actuator_ids(model, names)
    closed = _finger_close_closed_targets(
        names,
        model.actuator_ctrlrange[act_ids, 0],
        model.actuator_ctrlrange[act_ids, 1],
        mcp_target=profile.mcp_target,
        pip_target=profile.pip_target,
        dip_target=profile.dip_target,
    )
    start = targets[act_ids].copy()
    alpha = _smoothstep(time_seconds / max(profile.close_duration, 1e-6))
    targets[act_ids] = start + alpha * (closed - start)
    return np.clip(targets, low, high)


def grasp_target(
    model: mj.MjModel, time_seconds: float, profile: GraspProfile
) -> np.ndarray:
    """One actuator target vector for the requested pattern time (his logic).

    Structure, unchanged from his `grasp_target`: every pattern first runs the
    same smoothstep close from `pregrip` to `grip` over [CLOSE_START, CLOSE_END],
    with the thumb lagging by `thumb_delay`; only after `CLOSE_END + thumb_delay`
    does the pattern-specific modulation start. That shared prefix is why the
    per-pattern runs below all begin with the same ~2 s of settling contact.
    """
    validate_profile(profile)

    if profile.pattern == "finger_close":
        return finger_close_grasp_target(model, time_seconds, profile)

    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    span = high - low
    is_thumb = thumb_mask(model)
    grip = low + np.where(is_thumb, profile.thumb_grip_fraction, profile.grip_fraction) * span
    pregrip = pregrip_targets(model, profile.pregrip_fraction)

    close_span = CLOSE_END - CLOSE_START
    finger_phase = np.clip((time_seconds - CLOSE_START) / close_span, 0.0, 1.0)
    thumb_phase = np.clip(
        (time_seconds - CLOSE_START - profile.thumb_delay) / close_span, 0.0, 1.0
    )
    phase = np.where(is_thumb, thumb_phase, finger_phase)
    targets = pregrip + _smoothstep(phase) * (grip - pregrip)

    if time_seconds < CLOSE_END + profile.thumb_delay:
        return np.clip(targets, low, high)

    elapsed = time_seconds - CLOSE_END - profile.thumb_delay
    if profile.pattern == "hold":
        modulation = 0.0
    elif profile.pattern == "pulse":
        modulation = 0.5 * profile.pulse_amplitude * (
            1.0 - np.cos(2.0 * np.pi * profile.pulse_hz * elapsed)
        )
    elif profile.pattern == "squeeze":
        # Staircase up then back down, one tread per SQUEEZE_STEP_SECONDS.
        cycle = profile.squeeze_steps * 2
        index = int(elapsed / SQUEEZE_STEP_SECONDS) % cycle
        level = index if index < profile.squeeze_steps else cycle - index - 1
        modulation = profile.pulse_amplitude * level / max(profile.squeeze_steps - 1, 1)
    elif profile.pattern == "regrasp":
        release = 0.5 * (1.0 - np.cos(2.0 * np.pi * REGRASP_HZ * elapsed))
        return np.clip(targets + release * (pregrip - targets), low, high)
    elif profile.pattern == "tap":
        # Only TAP_DEPTH of the way back to pregrip, but at TAP_HZ — fast enough
        # that contacts break and re-form, which is the ncon variation we want.
        release = 0.5 * (1.0 - np.cos(2.0 * np.pi * TAP_HZ * elapsed))
        return np.clip(targets + TAP_DEPTH * release * (pregrip - targets), low, high)
    elif profile.pattern == "shear":
        lateral = profile.shear_amplitude * np.sin(2.0 * np.pi * SHEAR_HZ * elapsed)
        return np.clip(
            targets + np.where(lateral_mask(model), lateral * span, 0.0), low, high
        )
    else:
        raise ValueError(f"Unknown grasp pattern '{profile.pattern}'")

    return np.clip(targets + np.where(is_thumb, 0.0, modulation * span), low, high)


# --------------------------------------------------------------------------- #
# Driving our scene
# --------------------------------------------------------------------------- #

# Seconds of pattern time to simulate, chosen per pattern to cover a whole
# modulation cycle. Every pattern spends its first CLOSE_END + THUMB_DELAY = 2.0 s
# doing the shared close, so anything less than ~2.5 s samples only that.
PATTERN_SECONDS: dict[str, float] = {
    "finger_close": 3.0,   # close_duration 1.0 s, then hold
    "hold": 3.5,
    "pulse": 5.0,          # 1.0 Hz -> 3 cycles
    "squeeze": 9.5,        # 8 treads x 0.9 s = 7.2 s -> one full staircase
    "regrasp": 8.0,        # 0.35 Hz -> ~2 cycles
    "tap": 5.0,            # 1.5 Hz -> ~7 cycles
    "shear": 6.5,          # 0.7 Hz -> ~4.5 cycles
}

# Frames banked per trial. Measured: fixture coverage is driven far more by trial
# DIVERSITY (which patch of hand the cube is pressed against) than by temporal
# density within one trial — a single densely sampled trial saturates around
# 60-90 live taxels no matter how often it is sampled, while differently posed
# trials keep adding new ones. So: many short-sampled trials, not few dense ones.
# `tap` gets 2x because it is the pattern whose point is contact making and
# breaking; sampled too coarsely we alias its 1.5 Hz release and lose exactly the
# transitions through zero we want.
FRAMES_PER_TRIAL: dict[str, int] = {p: 9 for p in GRASP_PATTERNS} | {"tap": 18}

# Steps held at `pregrip` before pattern time starts, so the cube settles into
# the pre-grip pose instead of being hit by a transient at t=0.
SETTLE_STEPS = 300

# --- OUR ONE DELIBERATE DEVIATION FROM HIS CONSTANTS ----------------------- #
# His THUMB_GRIP_FRACTION = 0.35 is a fraction of the *ctrl range*, and our RL
# hand's thumb ranges are not his flex model's: 0.35 puts th_cmc/th_mcp at
# ~0.51/0.55 rad, which is more OPEN than the scene's own `home` keyframe pose
# (0.8 rad). The result, measured: the thumb never opposes the cube, and the
# 46 th_ds + 16 th_px thumb taxels record exactly zero contact in every pattern.
# 0.70 is the smallest fraction that produces sustained thumb-tip contact
# without ejecting the cube. `thumb_grip_fraction` is a field of HIS
# `GraspProfile`, so this is a profile instantiation, not a change to his logic;
# half the trials below still run his default profile verbatim.
THUMB_OPPOSED_FRACTION = 0.70

# Cube start-pose jitter, in the cube's own home frame. The single scripted pose
# only ever presents one cube face to one patch of palm; jittering it is what
# actually moves the contact patch around the hand, and it is a scene choice
# rather than a motion-logic change.
CUBE_POS_JITTER_LO = np.array([-0.020, -0.030, -0.015])
CUBE_POS_JITTER_HI = np.array([+0.030, +0.030, +0.020])


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion in MuJoCo (w, x, y, z) order."""
    u = rng.random(3)
    x, y = np.sqrt(1.0 - u[0]), np.sqrt(u[0])
    return np.array([
        y * np.cos(2 * np.pi * u[2]),   # w
        x * np.sin(2 * np.pi * u[1]),
        x * np.cos(2 * np.pi * u[1]),
        y * np.sin(2 * np.pi * u[2]),
    ])


@dataclass(frozen=True)
class Trial:
    pattern: str
    profile: GraspProfile
    seed: int          # 0 = the scene's own `home` cube pose, no jitter
    label: str


N_DEFAULT_TRIALS = 4    # seed 0 is the scene's own home pose, 1..3 are jittered
N_OPPOSED_TRIALS = 4


def trial_matrix() -> list[Trial]:
    """(pattern x cube pose x profile) combinations banked into the fixture.

    Eight trials per pattern:
      seed 0        his default profile, the scene's unmodified `home` cube pose
                    — the faithful baseline, directly comparable to what his
                    script produces on his own model;
      seeds 1-3     his default profile, jittered cube pose — same motion,
                    different patch of hand touched;
      seeds 4-7     thumb-opposed profile, jittered cube pose — the only trials
                    in which the thumb taxels see anything at all.

    Whether a given (pose, profile) pair ends in a held cube or a dropped one is
    left to the physics; dropped trials are kept, because a hand that has just
    lost the cube is exactly where the ncon == 0 frames come from.
    """
    trials = []
    for pattern in GRASP_PATTERNS:
        default = GraspProfile(pattern=pattern)
        opposed = GraspProfile(pattern=pattern, thumb_grip_fraction=THUMB_OPPOSED_FRACTION)
        for s in range(N_DEFAULT_TRIALS):
            label = "his default / home pose" if s == 0 else f"his default / pose {s}"
            trials.append(Trial(pattern, default, seed=s, label=label))
        for s in range(N_DEFAULT_TRIALS, N_DEFAULT_TRIALS + N_OPPOSED_TRIALS):
            trials.append(Trial(pattern, opposed, seed=s, label=f"thumb-opposed / pose {s}"))
    return trials


def run_trial(model, data, sensor, obj_geoms, trial: Trial, *, verbose=True):
    """Simulate one trial from the `home` keyframe; return the sampled frames.

    The scene is reset to `home` each time so trials are independent — a cube
    knocked out of the hand by `tap` must not silently starve the next trial of
    contacts.
    """
    profile = trial.profile
    validate_profile(profile)

    mj.mj_resetDataKeyframe(model, data, 0)
    if trial.seed:
        # qpos[16:19] / qpos[19:23] are the cube freejoint's position and quat;
        # the hand is pinned, so this is the only pose freedom in the scene.
        rng = np.random.default_rng(trial.seed * 1000 + GRASP_PATTERNS.index(trial.pattern))
        data.qpos[16:19] += rng.uniform(CUBE_POS_JITTER_LO, CUBE_POS_JITTER_HI)
        data.qpos[19:23] = _random_quat(rng)

    data.ctrl[:] = grasp_target(model, 0.0, profile)
    for _ in range(SETTLE_STEPS):
        mj.mj_step(model, data)

    t0 = data.time
    n_steps = int(round(PATTERN_SECONDS[trial.pattern] / model.opt.timestep))
    every = max(1, n_steps // FRAMES_PER_TRIAL[trial.pattern])
    frames = []
    for step in range(n_steps):
        data.ctrl[:] = grasp_target(model, data.time - t0, profile)
        mj.mj_step(model, data)
        taxels = sensor.update(data)
        if step % every == 0:
            frames.append(REF.capture(model, data, taxels, obj_geoms))

    if verbose:
        tax = np.stack([f["taxels"] for f in frames])
        mag = np.linalg.norm(tax, axis=2)
        ncons = np.asarray([f["ncon"] for f in frames])
        print(
            f"  {trial.pattern:<13}{trial.label:<32}{len(frames):>4} fr  "
            f"ncon {ncons.min():>2}/{int(np.median(ncons)):>2}/{ncons.max():>2}  "
            f"zero {int((ncons == 0).sum()):>2}  "
            f"taxels {(mag > 1e-6).any(axis=0).sum():>3}/368  "
            f"|shear| {np.abs(tax[:, :, :2]).max():.3f}  "
            f"|normal| {np.abs(tax[:, :, 2]).max():.3f}"
        )
    return frames


# --------------------------------------------------------------------------- #
# Coverage reporting
# --------------------------------------------------------------------------- #


def coverage(frames: list[dict], layout) -> dict:
    """Everything we claim about a fixture, computed from the fixture itself."""
    tax = np.stack([f["taxels"] for f in frames])          # (F, 368, 3)
    ncon = np.asarray([f["ncon"] for f in frames])
    mag = np.linalg.norm(tax, axis=2)
    ever = (mag > 1e-6).any(axis=0)

    per_body = {}
    for src in BODY_MAP:
        idx = [i for i, e in enumerate(layout.entries) if e.body == src]
        per_body[src] = (int(ever[idx].sum()), len(idx))

    return dict(
        n_frames=len(frames),
        n_active=int(ever.sum()),
        per_body=per_body,
        ncon_min=int(ncon.min()),
        ncon_med=float(np.median(ncon)),
        ncon_max=int(ncon.max()),
        ncon_zero=int((ncon == 0).sum()),
        chan=[(float(tax[:, :, k].min()), float(tax[:, :, k].max())) for k in range(3)],
    )


def print_coverage(title: str, cov: dict) -> None:
    print(f"\n{title}")
    print(f"  frames {cov['n_frames']},  taxels ever active {cov['n_active']}/368")
    for src, (n, total) in cov["per_body"].items():
        flag = "  <- fingertip (placement unverified, taxel_map.SUSPECT)" if src in SUSPECT else ""
        print(f"    {BODY_MAP[src]:<7} ({src:<24}) {n:>3}/{total:<3}{flag}")
    print(
        f"  contacts per frame: min {cov['ncon_min']}  median {cov['ncon_med']:.1f}  "
        f"max {cov['ncon_max']};  ncon==0 frames: {cov['ncon_zero']}"
    )
    for k, name in enumerate(("shear_x", "shear_y", "normal_z")):
        lo, hi = cov["chan"][k]
        print(f"    {name:<9} {lo:+.4f} .. {hi:+.4f}")


def load_fixture_frames(path: pathlib.Path) -> list[dict]:
    """Re-read a banked fixture into the same dict-per-frame form as `capture`."""
    npz = np.load(path)
    n = int(npz["n_frames"])
    return [
        dict(ncon=int(npz[f"f{i:03d}_ncon"]), taxels=npz[f"f{i:03d}_taxels"])
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #


def main() -> None:
    layout = build_layout()
    model = REF.build_model()
    data = mj.MjData(model)
    obj_geoms = REF.cube_geom_names(model)

    sensor = VirtualTaxelSensor(
        model, remap_layout(layout), obj_geoms, KERNEL_SIGMA, KERNEL_CUTOFF
    )

    print("=" * 78)
    print("GRASP PATTERNS (ported from mohammad200h/sparsh-skin-sim util/motion_util.py)")
    print("=" * 78)
    print(f"  scene: {REF.SCENE.name},  dt={model.opt.timestep}, nu={model.nu}")
    print(f"  cube geoms: {obj_geoms}")
    print(f"  shear drives: "
          f"{[n for n, m in zip(actuator_names(model), lateral_mask(model)) if m]}\n")

    all_frames: list[dict] = []
    for trial in trial_matrix():
        all_frames += run_trial(model, data, sensor, obj_geoms, trial)

    OUT.mkdir(exist_ok=True)
    path = OUT / "oracle_fixture_v2.npz"
    REF.save_fixture(path, all_frames)

    print("\n" + "=" * 78)
    print("COVERAGE")
    print("=" * 78)
    old = OUT / "oracle_fixture.npz"
    if old.exists():
        print_coverage(f"BEFORE — {old.name} (all actuators to 60% of range)",
                       coverage(load_fixture_frames(old), layout))
    n_trials = len(trial_matrix())
    print_coverage(f"AFTER — {path.name} ({len(GRASP_PATTERNS)} grasp patterns x "
                   f"{n_trials // len(GRASP_PATTERNS)} trials)",
                   coverage(all_frames, layout))

    print(f"\nextended oracle fixture -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(all_frames)} frames across {len(trial_matrix())} trials, "
          f"ncon==0 frames kept (they are a genuine edge case)")
    print(f"  {old.name} left untouched")


if __name__ == "__main__":
    main()
