"""Single source of truth for the tactile->RL body mapping, plus shared helpers.

Derived and validated in 02_map_bodies.py. See PROJECT_LOG.md §1.5b.

The taxel layout (leapxela) and the model mjlab trains on are two independent
CAD lineages with zero body-name overlap. This module holds the correspondence
so no script re-derives it — a wrong map produces plausible numbers attributed
to the wrong finger, which is the failure mode we are guarding against.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco as mj
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = _ROOT / "third_party" / "leapXELA_model"                      # current main
PINNED_DIR = (                                                            # what mjlab loads
    _ROOT / "third_party" / "leapXelaMjLab" / "src" / "leap_xela_mjlab"
    / "assets" / "leapXELA_model"
)

TACTILE_XML = MODEL_DIR / "leapxela" / "leapXela_pointcloud" / "robot.xml"

# The RIGID hand mjlab actually trains on (submodule pinned at 4e8003f):
# 17 bodies, 25 joints, 0 connects, 0 taxel sites. This is the splat encoder's
# target. Do NOT substitute current main's file of the same name -- that one is
# the flex model: 385 bodies, 1129 joints, 368 connects, and its 368 taxels sit
# one-per-flex-body, which breaks the splat's group-by-body assumption.
# See PROJECT_LOG.md §1.5b.
RL_XML = PINNED_DIR / "leapXela_generated_mjx_Box.xml"
FLEX_XML = MODEL_DIR / "leapXela_generated_mjx_Box.xml"

# WARNING: chain order in the tactile lineage is RING, MIDDLE, INDEX, THUMB.
# `finger` is the RING finger. Mapping by name order silently swaps index/ring.
BODY_MAP: dict[str, str] = {
    "leap_hand_xela_back_cover": "palm",
    "p3_unified": "rf_px",   "p2_unified": "rf_md",   "finger": "rf_ds",
    "p3_unified_2": "mf_px", "p2_unified_2": "mf_md", "finger_2": "mf_ds",
    "p3_unified_3": "if_px", "p2_unified_3": "if_md", "finger_3": "if_ds",
    "thp1_unified": "th_px", "thumb": "th_ds",
}

# Bodies whose local mesh bounding boxes match to 0.000 mm: taxels transfer
# with no transform.
CONFIRMED = {
    "leap_hand_xela_back_cover",
    "p3_unified", "p2_unified",
    "p3_unified_2", "p2_unified_2",
    "p3_unified_3", "p2_unified_3",
    "thp1_unified",
}

# Fingertips: the mounting frame agrees but the RL tip is a physically
# different, ~10 mm longer part in every mjx variant. Taxels placed here sit
# inside the RL tip rather than on its surface. Unresolved — question for Hamid.
SUSPECT = {"finger", "finger_2", "finger_3", "thumb"}

assert set(BODY_MAP) == CONFIRMED | SUSPECT


def remap_layout(layout):
    """Return a copy of the layout with `entry.body` renamed to RL-lineage names.

    Injecting the sites is NOT sufficient to run `VirtualTaxelSensor` against the
    mjlab model. Its constructor groups taxels by body with

        body_id = mj_name2id(model, mjOBJ_BODY, entry.body)

    and `entry.body` holds tactile-lineage names ('finger', 'p2_unified', ...),
    which do not exist in the RL model -- every lookup returns -1, every taxel
    lands in one bogus group, and no contact ever matches. The encoder runs
    happily and outputs all zeros.

    Remapping the layout rather than patching the encoder keeps the reference
    implementation byte-for-byte untouched, which matters because it is the
    correctness oracle.
    """
    import dataclasses

    entries = tuple(
        dataclasses.replace(e, body=BODY_MAP[e.body]) for e in layout.entries
    )
    return dataclasses.replace(layout, entries=entries)


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate row-vectors v (N,3) by MuJoCo quaternion q (w,x,y,z)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return v @ R.T


@dataclass
class Shape:
    nvert: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    centroid: np.ndarray
    verts: np.ndarray | None = None

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def body_shapes(model: mj.MjModel, keep_verts: bool = False) -> dict[str, Shape]:
    """Per body: all mesh vertices expressed in that body's own local frame."""
    out: dict[str, Shape] = {}
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        pts = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid:
                continue
            if model.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[gid]
            if mid < 0:
                continue
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[adr : adr + num].reshape(-1, 3)
            pts.append(quat_rot(model.geom_quat[gid], v) + model.geom_pos[gid])
        if pts:
            allv = np.concatenate(pts)
            out[name] = Shape(
                nvert=len(allv),
                bbox_min=allv.min(0),
                bbox_max=allv.max(0),
                centroid=allv.mean(0),
                verts=allv if keep_verts else None,
            )
    return out
