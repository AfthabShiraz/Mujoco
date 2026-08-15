"""Tactile observations for `leapXelaMjLab`: taxel sites, and the term that reads them.

This is the piece that turns a fast encoder into a feature. `benchmarks/harness.py`
measures the splat encoder against a bare `mjw.step` loop; nothing there touches a
policy, a reward or an episode. This module puts the same encoder inside the
supervisor's actual training environment, as an `ObservationTermCfg` alongside
`joint_pos`, `cube_pos_error` and the rest.

Two things have to happen, and both are places where a mistake is silent rather
than loud:

1. THE HAND MODEL HAS NO TAXELS IN IT. `leapXela_generated_mjx_Box.xml` -- the
   file `leap_xela.get_hand_xml()` returns -- contains zero `taxel_*` sites
   (PROJECT_LOG §1.5b). `entity_cfg_with_taxels` wraps the supervisor's own
   `get_spec()` and injects all 368 through `MjSpec` before mjlab compiles it,
   which is the same mechanism `explore/03_inject_taxels.py` validated and the
   benchmark harness uses. Nothing about the robot's dynamics changes: sites are
   massless and collisionless.

2. MJLAB NAMESPACES EVERYTHING. Once mjlab assembles the scene, bodies are
   `robot/palm`, taxel sites are `robot/taxel_017`, and the manipulated cube's
   geom is `cube/cube`. The taxel layout ships tactile-lineage body names, which
   `taxel_map.remap_layout` converts to RL-lineage ones (`palm`, `if_px`) -- but
   those still are not what mjlab compiled. Both fixes are needed, and they
   compose. `prepare_static` raises on a failed lookup rather than emitting
   zeros, so getting this wrong stops the run instead of quietly producing a
   policy trained on an all-zero tactile channel.

WHAT THE TERM RETURNS. `(num_envs, 1104)` -- 368 taxels x 3 channels
(`shear_x`, `shear_y`, `normal_z`), flattened, float32, already on the GPU. It is
the same array the CPU reference `VirtualTaxelSensor` produces, validated against
it to 4e-07 in float64 (`tests/test_encoder_oracle.py`).

ORDERING. The term reads `env.sim.wp_data` *after* mjlab has stepped physics for
this control step, so the contacts it encodes are the ones the policy is about to
observe. mjlab evaluates observation terms after `sim.step()`, so no explicit
synchronisation is needed here beyond keeping Warp on torch's stream, which
`SplatEncoder` handles.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Repo root: this file is <root>/src/mjlab_tactile/taxel_term.py
ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "benchmarks", ROOT / "src", ROOT / "explore",
           ROOT / "third_party" / "leapXELA_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SITE_SIZE = 0.0012          # matches benchmarks/harness.py
ENTITY_PREFIX = "robot/"    # how mjlab namespaces the hand entity
CUBE_GEOM = "cube/cube"     # the manipulated object, as mjlab names it


# --------------------------------------------------------------------------- #
# 1. Model side: inject the taxel sites
# --------------------------------------------------------------------------- #


def inject_taxel_sites(spec) -> int:
    """Add the 368 taxel sites to an mjlab hand `MjSpec`, in place.

    Returns the number injected. Site names are the layout's own
    (`taxel_000` ...); mjlab prepends `robot/` when it compiles the scene.
    """
    from leapxela.taxel_layout import build_layout
    from taxel_map import BODY_MAP

    bodies = {b.name: b for b in spec.bodies}
    layout = build_layout()
    n = 0
    for e in layout.entries:
        target = BODY_MAP[e.body]
        if target not in bodies:
            raise KeyError(
                f"body '{target}' (mapped from '{e.body}') is absent from the hand "
                f"spec; BODY_MAP is out of date with the model mjlab loads"
            )
        bodies[target].add_site(
            name=e.site_name,
            pos=np.asarray(e.pos, dtype=float),
            quat=np.asarray(e.quat, dtype=float),
            size=[SITE_SIZE] * 3,
            group=4,
        )
        n += 1
    return n


def entity_cfg_with_taxels(finger_tip_type: str = "Box"):
    """The supervisor's `EntityCfg` for the hand, with taxel sites added.

    Built by wrapping his `get_spec` rather than copying it, so any change he
    makes to the hand -- assets, actuators, defaults -- carries through here
    untouched. The only difference is 368 extra sites.
    """
    import dataclasses

    from leap_xela_mjlab.robots.leap_xela import get_leap_xela_cfg, get_spec

    base = get_leap_xela_cfg(finger_tip_type)

    def _spec_fn():
        spec = get_spec(finger_tip_type)
        inject_taxel_sites(spec)
        return spec

    return dataclasses.replace(base, spec_fn=_spec_fn)


# --------------------------------------------------------------------------- #
# 1b. An mjlab inefficiency that adding sites turns from small into large
# --------------------------------------------------------------------------- #


def patch_mjlab_site_pos_w() -> bool:
    """Make `EntityData.site_pos_w` read positions instead of deriving them.

    NOT a tactile fix, and not our bug. As shipped (mjlab 1.5.3,
    `entity/data.py`):

        @property
        def site_pos_w(self): return self.site_pose_w[..., 0:3]

        @property
        def site_pose_w(self):
            pos_w = self.data.site_xpos[:, self.indexing.site_ids]
            mat_w = self.data.site_xmat[:, self.indexing.site_ids]
            quat_w = quat_from_matrix(mat_w)
            return torch.cat([pos_w, quat_w], dim=-1)

    So asking for **one** site's position gathers every site's position *and*
    3x3 orientation, runs `quat_from_matrix` over all of them, allocates an
    `(num_envs, num_sites, 7)` tensor, and then slices three columns out and
    discards the quaternions it just computed.

    The reorient task does exactly that twice per step:
    `_palm_pos_w` (one site, feeding the actor's `cube_pos_error`) and
    `fingertip_positions_rel_palm` (four sites, critic).

    With the stock hand's 68 sites this is wasteful but cheap. Injecting 368
    taxel sites makes it 436 sites, and the waste becomes the single largest
    cost in the environment. Measured at N=2048, taxel hand:

        as shipped                    49.92 ms/env-step
        site_pos_w -> site_xpos       15.07 ms/env-step    (3.31x)

    For comparison the plain 68-site hand runs at 16.61 ms, so with the patch
    the taxel hand is *cheaper per step than the stock hand is without it* --
    the stock hand pays the same wasted work, just 6x less of it.

    This is a genuine upstream bug worth reporting to mjlab, and it is worth
    separating from anything about tactile sensing: any mjlab user who adds
    sites to a robot pays it on every unrelated site query.

    Returns True if the patch was applied, False if it was already applied.
    """
    from mjlab.entity.data import EntityData

    if getattr(EntityData, "_taxel_site_pos_patched", False):
        return False
    EntityData.site_pos_w = property(
        lambda self: self.data.site_xpos[:, self.indexing.site_ids]
    )
    EntityData._taxel_site_pos_patched = True
    return True


# --------------------------------------------------------------------------- #
# 2. Observation side: the term
# --------------------------------------------------------------------------- #


class _EncoderCache:
    """Holds the encoder, built on first use and reused thereafter.

    `SplatEncoder` allocates its scratch against a fixed world count and takes
    zero-copy views of the Warp arrays, so it must be built after mjlab has
    created the simulation and must not outlive it. An observation term has no
    setup hook, so construction is deferred to the first call and keyed on the
    `Data` object's identity -- if mjlab ever reallocates, the encoder is rebuilt
    rather than left pointing at freed memory.
    """

    def __init__(self) -> None:
        self.enc = None
        self._key = None

    def get(self, env, encoder: str):
        import torch
        import warp as wp
        import mujoco_warp as mjw
        from harness import SplatEncoder, _attach_mjw

        sim = env.sim
        key = (id(sim.wp_data), env.num_envs)
        if self.enc is not None and self._key == key:
            return self.enc

        # DO NOT call `wp.set_stream` here. The benchmark harness does exactly
        # that -- it owns its Warp context, so moving Warp onto torch's stream
        # is the cheapest way to order `mjw.step` against the encoder. Inside
        # mjlab it is a segfault: `Simulation.__init__` captures `step`,
        # `forward`, `reset` and `sense` into CUDA graphs (`sim.py:347`) bound to
        # the Warp stream that was current at capture time, and observation
        # terms are constructed *after* that. Repointing the stream underneath a
        # captured graph dumps core, which is how this was found.
        #
        # Ordering is handled in `taxel_forces` instead, by making torch's
        # stream wait on Warp's rather than by moving either of them.

        mjm = sim.mj_model
        per_env = max(1, int(sim.wp_data.naconmax) // max(1, env.num_envs))
        enc = SplatEncoder(
            mjm, sim.wp_model, sim.wp_data, env.num_envs,
            encoder=encoder,
            name_prefix=ENTITY_PREFIX,
            object_geom_names=[CUBE_GEOM],
            contacts_per_env=per_env,
        )
        self.enc = _attach_mjw(enc, mjw)
        self._key = key
        return self.enc


_CACHE = _EncoderCache()


def taxel_forces(env, encoder: str = "triton", flatten: bool = True):
    """Per-taxel contact force -> `(num_envs, 1104)`, or `(num_envs, 368, 3)`.

    `encoder` selects the implementation: `"triton"` is the two-pass gather
    kernel, `"torch"` the dense reference, `"compile"` the fused-eager one. All
    three are numerically equivalent (`tests/test_triton_kernel.py`); the default
    is the kernel because the dense version costs 4.6x throughput at scale and
    there is no reason to pay that inside a training loop.
    """
    import torch
    import warp as wp

    enc = _CACHE.get(env, encoder)

    # Order torch behind Warp without moving either stream (see `_EncoderCache`).
    # mjlab has just launched the physics graph on Warp's stream; the encoder is
    # torch work reading the contact arrays that graph writes. Without this the
    # encoder can read a half-written contact list -- which would not crash, it
    # would quietly produce taxel values from a torn frame.
    wp_stream = wp.get_stream()
    if wp_stream is not None and torch.cuda.is_available():
        torch.cuda.current_stream().wait_stream(wp.stream_to_torch(wp_stream))

    out = enc.encode()                      # (B, 368, 3), on GPU
    return out.reshape(env.num_envs, -1) if flatten else out
