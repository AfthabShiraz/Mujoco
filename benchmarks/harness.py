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

Two representations:

  --tactile none    bare physics. The H1 control. Unchanged from the original
                    harness, so previously banked numbers stay comparable.

  --tactile splat   physics + the naive Gaussian-splat taxel encoder
                    (`src/encoders/taxel_torch.py`) evaluated every step for
                    every world. This is what H2 is about: at what env count
                    does touch cost more than the physics it observes?

The splat path is measured TWICE per run, on purpose:
  * a fused loop with no inner synchronisation -> the honest end-to-end
    throughput, directly comparable to the `none` numbers;
  * a second loop that synchronises between `mjw.step` and the encoder ->
    the physics/encoder split. Those inner syncs serialise the pipeline and
    cost a little, which is why they are not used for the headline figure.
    `split_ms_per_step` is reported alongside so the overhead is visible.

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

DEFAULT_SCENE = (
    ROOT / "third_party" / "leapXelaMjLab" / "src" / "leap_xela_mjlab"
    / "assets" / "leapXELA_model" / "scene_mjx_cube.xml"
)

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
CLOSE_FRACTION = 0.6
SETTLE_STEPS = 250   # CPU run shows first cube contact ~step 120

# VirtualTaxelSensor defaults, from leapxela/visualize_taxel_layout.py. The
# oracle in explore/out/oracle_fixture.npz was banked with these.
KERNEL_SIGMA = 0.0035
KERNEL_CUTOFF = 0.01
SITE_SIZE = 0.0012


def peak_rss_gb() -> float:
    """Peak resident set of this process. On unified memory this tracks the
    GPU allocation too, which is exactly the number we care about here."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_scene_model(scene: pathlib.Path, tactile: str) -> mj.MjModel:
    """Compile the benchmark scene, with taxel sites for the tactile paths.

    `none` compiles the scene untouched so its numbers stay comparable with the
    already-banked physics-only sweep. `splat` injects the 368 taxel sites via
    MjSpec exactly as `explore/03_inject_taxels.py` does -- the sites must exist
    in the model handed to `mjw.put_model`, because that is what allocates
    `d.site_xpos` / `d.site_xmat`.

    Sites are massless and collisionless, so they do not change the dynamics;
    they do add 368 site poses to the forward kinematics, and that cost lands in
    the *physics* half of the split. That is the honest place for it -- it is a
    cost the tactile representation imposes.
    """
    if tactile == "none":
        return mj.MjModel.from_xml_path(str(scene))

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

    def __init__(self, mjm: mj.MjModel, m, d, nworld: int):
        import torch
        import warp as wp

        from encoders.taxel_torch import N_TAXELS, prepare_static
        from leapxela.taxel_layout import build_layout
        from taxel_map import remap_layout

        self.torch = torch
        self.wp = wp
        self.m, self.d = m, d
        self.B = nworld
        self.C = NCONMAX_PER_ENV
        self.T = N_TAXELS
        self.naconmax = int(d.naconmax)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev

        # Layout body names are tactile-lineage; the model is RL-lineage. Skip
        # the remap and every lookup returns -1, all 368 taxels collapse into
        # one bogus group and the encoder outputs zeros without erroring
        # (PROJECT_LOG §1.5d). This is the single easiest way to get a fast
        # wrong answer here.
        self.static = prepare_static(mjm, remap_layout(build_layout()), cube_geom_names(mjm))

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
        from encoders.taxel_torch import encode_taxels

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
# Correctness gate
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--scene", type=pathlib.Path, default=DEFAULT_SCENE)
    ap.add_argument("--tactile", choices=["none", "splat"], default="none",
                    help="representation under test: 'none' is the H1 control")
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

    mjm = build_scene_model(args.scene, tactile)
    mjd = mj.MjData(mjm)
    mj.mj_forward(mjm, mjd)

    t0 = time.perf_counter()
    m = mjw.put_model(mjm)
    d = mjw.put_data(
        mjm, mjd, nworld=n,
        naconmax=NCONMAX_PER_ENV * n,   # total across worlds
        njmax=NJMAX_PER_ENV,            # per world -- do not scale
    )
    wp.synchronize()
    setup_s = time.perf_counter() - t0
    rss_after_alloc = peak_rss_gb()

    enc = None
    if tactile == "splat":
        # Warp and torch default to different CUDA streams. The fused timing
        # loop has no sync between `mjw.step` and the torch encoder, so without
        # this the encoder could read contact arrays the physics kernels have
        # not finished writing. Put Warp on torch's stream and the ordering is
        # the stream's problem, not ours.
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()), wp.get_device())
        enc = _attach_mjw(SplatEncoder(mjm, m, d, n), mjw)
        torch.cuda.reset_peak_memory_stats()

    # ---- close the hand so contacts exist ----------------------------------
    lo = mjm.actuator_ctrlrange[:, 0]
    hi = mjm.actuator_ctrlrange[:, 1]
    target = lo + CLOSE_FRACTION * (hi - lo)
    d.ctrl.assign(np.tile(target.astype(np.float32), (n, 1)))
    for _ in range(SETTLE_STEPS):
        mjw.step(m, d)
    wp.synchronize()

    # ---- correctness gate: no timing is worth anything if the answer is wrong
    checks = None
    if args.verify:
        if enc is None:
            print("[harness] --verify only applies to --tactile splat", file=sys.stderr)
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
    t0 = time.perf_counter()
    for _ in range(args.warmup):
        mjw.step(m, d)
        if enc is not None:
            enc.encode()
    wp.synchronize()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0

    contacts = int(d.nacon.numpy()[0])
    overflow = int((d.overflow.numpy() != 0).sum())

    # ---- timed, pass 1: fused, no inner syncs -> headline throughput --------
    wp.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        mjw.step(m, d)
        if enc is not None:
            enc.encode()
    wp.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # ---- timed, pass 2: synchronised between stages -> the split ------------
    # The inner syncs drain the pipeline twice per step, so `split_ms_per_step`
    # is an upper bound on the fused cost. Reported so the gap is visible rather
    # than quietly folded into one of the two halves.
    if enc is None:
        physics_s, encoder_s = elapsed, 0.0
    else:
        physics_s = encoder_s = 0.0
        for _ in range(args.steps):
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
        nconmax_per_env=NCONMAX_PER_ENV,
        njmax_per_env=NJMAX_PER_ENV,
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
        ok=True,
    )
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
