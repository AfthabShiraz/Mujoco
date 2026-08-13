"""Batched PyTorch port of `VirtualTaxelSensor.update` — the unoptimised baseline.

WHY this exists: the reference encoder (`leapxela/touch_sensor.py`) is a Python
loop over `data.ncon`, one environment at a time. On the RL side we need the
same 368x3 tactile map for thousands of environments per physics step, so the
first thing to establish is a version that is (a) *batched* over environments
and contacts and (b) provably identical to the reference. This module is that
version. It is deliberately written the obvious way — dense (B, C, T) tensors,
plain `einsum`, no fused kernels, no sparsity — because its cost is the number
every later implementation (C++/OpenMP/Triton/CUDA) is measured against. Making
it fast here would destroy the baseline.

Semantics reproduced exactly from the reference, in order:
  * the object-vs-hand geom test (which of the two contact geoms is the cube),
    and the resulting force sign flip — see `contact_inputs_from_geoms`;
  * per-contact Gaussian splat on the *in-plane* distance only;
  * hard zeroing beyond `kernel_cutoff` in-plane (`dist_sq > cutoff^2`) AND
    out-of-plane (`|local_z| > cutoff`, the patch-locality gate — PROJECT_LOG
    §1.5c, this one is load-bearing and geometry-dependent);
  * `weights /= weights.sum()` per contact, and skipping the contact entirely
    when that sum is <= 0;
  * restricting each contact's splat to the taxels of the one body it touched;
  * `out[:, 2] *= -1` at the end (compression positive).

The reference accumulates in float64; this runs in the input dtype (float32 in
practice), so agreement is to float32 rounding, not bit-exact. Measured error
against `explore/out/oracle_fixture.npz` is reported by
`tests/test_encoder_oracle.py`.

Two entry points:
  * `prepare_static(...)` — everything that needs the MuJoCo model (geom->body
    lookups, taxel->body grouping, taxel site ids). Done once, off the hot path.
  * `encode_taxels(...)` — pure tensor work, no model, no numpy, no Python loop
    over environments or contacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")  # headless box; must precede mujoco import

import numpy as np
import torch

N_TAXELS = 368


# --------------------------------------------------------------------------- #
# Static preparation (needs the model; run once)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaxelStatic:
    """Model-derived lookups the batched encoder needs but must not re-derive.

    `body_slot` is a dense 0..n_hand_bodies-1 relabelling of the 12 taxel-carrying
    MuJoCo body ids. The encoder only ever compares body labels for equality, so
    the relabelling is free and keeps `taxel_body` small and contiguous.
    """

    site_ids: np.ndarray        # (368,)   index into data.site_xpos / site_xmat
    taxel_body: np.ndarray      # (368,)   body slot of each taxel
    geom_body_slot: np.ndarray  # (ngeom,) body slot of a geom, -1 if not a taxel body
    geom_is_object: np.ndarray  # (ngeom,) bool, geom belongs to the manipulated object
    body_ids: np.ndarray        # (n_hand_bodies,) slot -> MuJoCo body id


def prepare_static(model, layout, object_geom_names) -> TaxelStatic:
    """Mirror `VirtualTaxelSensor.__init__` as flat arrays.

    `layout` must already be remapped to RL-lineage body names via
    `taxel_map.remap_layout` — otherwise every `mj_name2id` returns -1, all 368
    taxels collapse into one bogus group and the encoder silently outputs zeros
    (PROJECT_LOG §1.5d).
    """
    import mujoco as mj

    site_ids = np.array(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, e.site_name) for e in layout.entries],
        dtype=np.int64,
    )
    if np.any(site_ids < 0):
        raise ValueError("Scene model is missing taxel sites; inject them first")

    # Taxel -> body id, in flat taxel order, then densify the body ids to slots.
    taxel_body_id = np.array(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, e.body) for e in layout.entries],
        dtype=np.int64,
    )
    if np.any(taxel_body_id < 0):
        raise ValueError("Layout body names absent from the model; use remap_layout()")
    body_ids = np.unique(taxel_body_id)
    taxel_body = np.searchsorted(body_ids, taxel_body_id)

    # geom -> slot of its body (-1 when that body carries no taxels), matching the
    # reference's `body2 in self._body_taxels` membership test.
    geom_body_slot = np.full(model.ngeom, -1, dtype=np.int64)
    for gid in range(model.ngeom):
        hit = np.searchsorted(body_ids, model.geom_bodyid[gid])
        if hit < len(body_ids) and body_ids[hit] == model.geom_bodyid[gid]:
            geom_body_slot[gid] = hit

    geom_is_object = np.zeros(model.ngeom, dtype=bool)
    for name in object_geom_names:
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"Contact geom '{name}' not found in model")
        geom_is_object[gid] = True

    return TaxelStatic(
        site_ids=site_ids,
        taxel_body=taxel_body,
        geom_body_slot=geom_body_slot,
        geom_is_object=geom_is_object,
        body_ids=body_ids,
    )


def contact_inputs_from_geoms(static: TaxelStatic, geom1, geom2):
    """Vectorised form of the reference's object-vs-hand test and sign flip.

    Returns `(contact_body, contact_sign, contact_valid)` for arbitrarily shaped
    `geom1`/`geom2` (numpy int arrays). The reference is:

        if geom1 is object and body(geom2) carries taxels:  hand = body2, sign = +1
        elif geom2 is object and body(geom1) carries taxels: hand = body1, sign = -1
        else: skip

    The asymmetry matters: the first branch wins ties, and the sign follows from
    contact.frame's normal pointing geom1 -> geom2, so the constraint force acts
    +frame^T f on geom2 and -frame^T f on geom1.

    This lives outside `encode_taxels` because it needs the model-derived geom
    lookups; the sign is then handed to the encoder as a plain tensor.
    """
    g1 = np.asarray(geom1, dtype=np.int64)
    g2 = np.asarray(geom2, dtype=np.int64)

    hand_is_geom2 = static.geom_is_object[g1] & (static.geom_body_slot[g2] >= 0)
    hand_is_geom1 = (
        ~hand_is_geom2 & static.geom_is_object[g2] & (static.geom_body_slot[g1] >= 0)
    )
    contact_valid = hand_is_geom2 | hand_is_geom1

    contact_body = np.where(hand_is_geom2, static.geom_body_slot[g2], static.geom_body_slot[g1])
    contact_body = np.where(contact_valid, contact_body, -1)
    contact_sign = np.where(hand_is_geom2, 1.0, -1.0)
    return contact_body, contact_sign, contact_valid


# --------------------------------------------------------------------------- #
# The encoder
# --------------------------------------------------------------------------- #


def encode_taxels(
    contact_pos: torch.Tensor,     # (B, C, 3)     world-frame contact points
    contact_frame: torch.Tensor,   # (B, C, 3, 3)  rows = (normal, tan1, tan2)
    contact_force: torch.Tensor,   # (B, C, 6)     mj_contactForce output, contact frame
    contact_sign: torch.Tensor,    # (B, C)        +1 hand is geom2, -1 hand is geom1
    contact_body: torch.Tensor,    # (B, C)        body slot touched, -1 = invalid/padding
    contact_valid: torch.Tensor,   # (B, C)        bool
    site_pos: torch.Tensor,        # (B, T, 3)     world-frame taxel positions
    site_mat: torch.Tensor,        # (B, T, 3, 3)  taxel local -> world
    taxel_body: torch.Tensor,      # (T,)          body slot of each taxel
    kernel_sigma: float,
    kernel_cutoff: float,
) -> torch.Tensor:
    """Per-taxel force (B, T, 3) in taxel-local frames, compression positive.

    Fully batched: no Python loop over environments, contacts or bodies. Runs on
    whatever device the inputs are on. Peak memory is O(B*C*T) — that is the
    point of the baseline, not an oversight.
    """
    B, C = contact_pos.shape[0], contact_pos.shape[1]
    T = site_pos.shape[1]
    dtype, device = contact_pos.dtype, contact_pos.device

    if C == 0:  # frames with ncon == 0 must give zeros, not NaN
        return torch.zeros((B, T, 3), dtype=dtype, device=device)

    # --- contact wrench into the world frame -------------------------------- #
    # force_world = frame^T @ f[:3], negated when the hand is geom1.
    force_world = torch.einsum("bcji,bcj->bci", contact_frame, contact_force[..., :3])
    force_world = force_world * contact_sign.unsqueeze(-1).to(dtype)

    # --- contact offset in every taxel's local frame ------------------------ #
    # rel[b, c, t] = contact_pos[b, c] - site_pos[b, t]
    rel = contact_pos.unsqueeze(2) - site_pos.unsqueeze(1)              # (B, C, T, 3)
    # site_mat maps local -> world, so v_local = R^T v_world.
    local = torch.einsum("btji,bctj->bcti", site_mat, rel)              # (B, C, T, 3)

    # --- splat weights ------------------------------------------------------ #
    dist_sq = local[..., 0] ** 2 + local[..., 1] ** 2                   # (B, C, T)
    weights = torch.exp(-dist_sq / (2.0 * kernel_sigma**2))

    # In-plane cutoff, then the out-of-plane patch-locality gate: taxels whose
    # plane is far from the contact point belong to a different patch on the
    # same body and must not receive any of this contact.
    weights = torch.where(dist_sq > kernel_cutoff**2, torch.zeros_like(weights), weights)
    weights = torch.where(
        local[..., 2].abs() > kernel_cutoff, torch.zeros_like(weights), weights
    )

    # Restrict each contact to the taxels of the body it actually touched, and
    # drop padded/invalid contacts. contact_body == -1 never matches a taxel.
    body_mask = contact_body.unsqueeze(-1) == taxel_body.view(1, 1, T)   # (B, C, T)
    keep = body_mask & contact_valid.unsqueeze(-1)
    weights = weights * keep.to(dtype)

    # --- per-contact normalisation ------------------------------------------ #
    # The reference skips a contact when its weight sum is <= 0. All weights are
    # >= 0, so weight_sum <= 0 implies every weight is already 0; dividing those
    # rows by 1 instead reproduces "skip" exactly without a branch.
    weight_sum = weights.sum(dim=-1, keepdim=True)                       # (B, C, 1)
    denom = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
    weights = weights / denom

    # --- accumulate --------------------------------------------------------- #
    # force_local[b, c, t] = site_mat[b, t]^T @ force_world[b, c]
    force_local = torch.einsum("btji,bcj->bcti", site_mat, force_world)  # (B, C, T, 3)
    out = (weights.unsqueeze(-1) * force_local).sum(dim=1)               # (B, T, 3)

    # Compression positive: pressing on a taxel pushes the hand along its -z.
    out = out * torch.tensor([1.0, 1.0, -1.0], dtype=dtype, device=device)
    return out
