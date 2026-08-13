"""Map the tactile lineage's bodies onto the RL lineage's bodies, and prove it.

Problem (PROJECT_LOG.md §1.5b): the taxel layout names bodies `finger`,
`p2_unified`, `leap_hand_xela_back_cover`; the model mjlab trains on names them
`rf_ds`, `rf_md`, `palm`. Zero name overlap. And taxel pos/quat are expressed in
the *tactile* lineage's body frames, so a name map alone is not enough — if the
two CAD exports place link frames differently, every taxel lands in the wrong
spot while still producing plausible-looking numbers.

Two things are established here:

1. WHICH body maps to which. Derived from the kinematic tree (both lineages are
   palm + 4 chains x 4 links) and disambiguated by the layout's own hardware
   patch names, NOT by guessing from the ordering. Guessing gets it wrong:
   `finger` is the RING finger, not the index.

2. WHETHER the frames agree. Each body's meshes are stored in body-local
   coordinates, so if two bodies are the same part exported with the same frame
   convention, their local vertex clouds share a bounding box and centroid.
   Vertex COUNTS differ (the RL lineage is mesh-simplified for MJX), so counts
   are deliberately ignored.

Run:
    PYTHONPATH=third_party/leapXELA_model .venv/bin/python explore/02_map_bodies.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import pathlib
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from leapxela.taxel_layout import build_layout

MD = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "leapXELA_model"
TACTILE_XML = MD / "leapxela" / "leapXela_pointcloud" / "robot.xml"
RL_CANDIDATES = [
    MD / "leapXela_generated_mjx_Box.xml",
    MD / "leapXela_generated_mjx.xml",
    MD / "leapXela_generated_mjx_CoACD.xml",
]

# Derived from the kinematic tree + PATCH_TO_HARDWARE. Chain order in the
# tactile lineage is RF, MF, IF, TH -- reversed from what the names suggest.
BODY_MAP = {
    "leap_hand_xela_back_cover": "palm",
    # ring finger
    "p3_unified": "rf_px",
    "p2_unified": "rf_md",
    "finger": "rf_ds",
    # middle finger
    "p3_unified_2": "mf_px",
    "p2_unified_2": "mf_md",
    "finger_2": "mf_ds",
    # index finger
    "p3_unified_3": "if_px",
    "p2_unified_3": "if_md",
    "finger_3": "if_ds",
    # thumb
    "thp1_unified": "th_px",
    "thumb": "th_ds",
}
# Carry no taxels, listed for completeness: p4_unified*->{rf,mf,if}_bs,
# thp2_unified->th_mp, clipper->th_bs.

TOL_MM = 1.0


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
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

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def body_shapes(model: mj.MjModel) -> dict[str, Shape]:
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
            out[name] = Shape(len(allv), allv.min(0), allv.max(0), allv.mean(0))
    return out


def main() -> None:
    layout = build_layout()
    taxels_per_body = {
        b: sum(1 for e in layout.entries if e.body == b) for b in {e.body for e in layout.entries}
    }
    missing = sorted(set(taxels_per_body) - set(BODY_MAP))
    if missing:
        raise SystemExit(f"BODY_MAP is incomplete, missing: {missing}")

    tac = body_shapes(mj.MjModel.from_xml_path(str(TACTILE_XML)))

    for rl_xml in RL_CANDIDATES:
        if not rl_xml.exists():
            continue
        rl = body_shapes(mj.MjModel.from_xml_path(str(rl_xml)))
        print("=" * 86)
        print(f"RL model: {rl_xml.name}")
        print("=" * 86)
        print(f"{'tactile body':<28}{'taxels':>7} {'->':^4}{'RL body':<10}"
              f"{'d_extent(mm)':>14}{'d_origin(mm)':>16}  verdict")
        print("-" * 86)

        agree = 0
        checked = 0
        for src_name, dst_name in BODY_MAP.items():
            n = taxels_per_body[src_name]
            if src_name not in tac or dst_name not in rl:
                print(f"{src_name:<28}{n:>7} {'->':^4}{dst_name:<10}{'(no mesh)':>30}")
                continue
            a, b = tac[src_name], rl[dst_name]
            # Compare BOUNDING BOXES, not centroids. The RL meshes are decimated,
            # so vertex density differs and the centroid shifts even when the
            # part and its frame are identical -- centroid gives false positives.
            d_min = (b.bbox_min - a.bbox_min) * 1000.0
            d_max = (b.bbox_max - a.bbox_max) * 1000.0
            d_ext = float(np.abs(a.extent - b.extent).max() * 1000.0)
            d_org = float(np.abs(np.minimum(np.abs(d_min), np.abs(d_max))).max())
            checked += 1
            if np.abs(d_min).max() < TOL_MM and np.abs(d_max).max() < TOL_MM:
                verdict = "FRAMES AGREE — taxels transfer unchanged"
                agree += 1
            elif d_org < TOL_MM:
                verdict = f"frame OK, part differs (+{d_ext:.1f}mm bigger)"
            else:
                verdict = "DIFFERENT GEOMETRY"
            print(f"{src_name:<28}{n:>7} {'->':^4}{dst_name:<10}"
                  f"{d_ext:>14.3f}{d_org:>16.3f}  {verdict}")

        print("-" * 86)
        covered = sum(taxels_per_body[s] for s in BODY_MAP)
        print(f"bodies whose frames agree: {agree}/{checked}   (taxels covered: {covered}/368)")
        print()


if __name__ == "__main__":
    main()
