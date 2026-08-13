"""Validate the batched torch encoder against the banked CPU oracle.

WHY: `src/encoders/taxel_torch.py` is a rewrite of the supervisor's
`VirtualTaxelSensor` — a different language, a different memory layout, and a
different loop structure. Every one of those is an opportunity to change the
answer while still producing plausible-looking forces on plausible-looking
taxels, which is the failure mode this whole line of work is guarding against
(PROJECT_LOG §1.5b/§1.5d). So the torch version is checked frame-by-frame
against `explore/out/oracle_fixture.npz`, which stores the reference encoder's
*inputs* alongside its 368x3 outputs — no MuJoCo stepping required, and hence no
exposure to the Warp-vs-CPU contact discrepancy.

Two things are checked, not one:
  1. per-frame (B=1) agreement with the oracle, including the ncon=0 frames,
     which must give exact zeros rather than NaN from a 0/0 normalisation;
  2. all 30 frames stacked into one padded B=30 batch giving the same answer as
     running them one at a time — the actual claim the batching makes.

The reference accumulates in float64 and casts to float32 at the end; this runs
in float32 throughout, so agreement is to float32 rounding. The test reports the
achieved error rather than pretending to a tolerance it derived from nothing,
and asserts only a loose bound well above the observed value.

Run (pytest is not installed in this venv yet, so run it directly; the file is
still a valid pytest module). Memory-capped as everything on this box must be —
unified memory means an oversized allocation takes down the host, not the
process (PROJECT_LOG §1.1):

    PYTHONPATH=third_party/leapXELA_model:explore \
      systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
      .venv/bin/python tests/test_encoder_oracle.py
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "explore", ROOT / "third_party" / "leapXELA_model"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from encoders.taxel_torch import (  # noqa: E402
    contact_inputs_from_geoms,
    encode_taxels,
    prepare_static,
)
from leapxela.taxel_layout import build_layout  # noqa: E402
from taxel_map import remap_layout  # noqa: E402

FIXTURE = ROOT / "explore" / "out" / "oracle_fixture.npz"
CHANNELS = ("shear_x", "shear_y", "normal_z")
ABS_TOL = 1e-4          # loose bound; the achieved error is reported below
BATCH_TOL = 1e-6        # per-frame vs batched: same maths, different padding


def _load_run_encoder():
    """Import explore/04_run_encoder.py by path — its name is not an identifier.

    Reusing its `build_model`/`cube_geom_names` rather than copying them is the
    point: the fixture was produced by *that* model, and a silently divergent
    rebuild here would invalidate the comparison.
    """
    path = ROOT / "explore" / "04_run_encoder.py"
    spec = importlib.util.spec_from_file_location("run_encoder_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_inputs():
    """Fixture + model -> per-frame numpy inputs for the batched encoder."""
    ref = _load_run_encoder()
    model = ref.build_model()
    layout = remap_layout(build_layout())
    static = prepare_static(model, layout, ref.cube_geom_names(model))

    npz = np.load(FIXTURE)
    n_frames = int(npz["n_frames"])
    frames = []
    for i in range(n_frames):
        p = f"f{i:03d}_"
        ncon = int(npz[p + "ncon"])
        geom1 = npz[p + "contact_geom1"].astype(np.int64)
        geom2 = npz[p + "contact_geom2"].astype(np.int64)
        body, sign, valid = contact_inputs_from_geoms(static, geom1, geom2)
        frames.append(
            dict(
                ncon=ncon,
                contact_pos=npz[p + "contact_pos"].reshape(ncon, 3),
                contact_frame=npz[p + "contact_frame"].reshape(ncon, 3, 3),
                contact_force=npz[p + "contact_force"].reshape(ncon, 6),
                contact_body=body,
                contact_sign=sign,
                contact_valid=valid,
                # site_xpos/site_xmat are stored for ALL model sites (436 of
                # them); the taxels are the 368 the layout named.
                site_pos=npz[p + "site_xpos"][static.site_ids],
                site_mat=npz[p + "site_xmat"][static.site_ids].reshape(-1, 3, 3),
                taxels=npz[p + "taxels"],
            )
        )
    return static, frames, float(npz["kernel_sigma"]), float(npz["kernel_cutoff"])


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)


def _encode_one(f, taxel_body, sigma, cutoff):
    """Single frame as a batch of one."""
    return encode_taxels(
        contact_pos=_t(f["contact_pos"]).unsqueeze(0),
        contact_frame=_t(f["contact_frame"]).unsqueeze(0),
        contact_force=_t(f["contact_force"]).unsqueeze(0),
        contact_sign=_t(f["contact_sign"]).unsqueeze(0),
        contact_body=_t(f["contact_body"], torch.int64).unsqueeze(0),
        contact_valid=_t(f["contact_valid"], torch.bool).unsqueeze(0),
        site_pos=_t(f["site_pos"]).unsqueeze(0),
        site_mat=_t(f["site_mat"]).unsqueeze(0),
        taxel_body=taxel_body,
        kernel_sigma=sigma,
        kernel_cutoff=cutoff,
    )[0]


def _encode_batched(frames, taxel_body, sigma, cutoff):
    """All frames as one padded (B, C, ...) batch, C = max ncon."""
    B = len(frames)
    C = max(1, max(f["ncon"] for f in frames))
    pos = np.zeros((B, C, 3))
    frame = np.zeros((B, C, 3, 3))
    force = np.zeros((B, C, 6))
    sign = np.zeros((B, C))
    body = np.full((B, C), -1, dtype=np.int64)
    valid = np.zeros((B, C), dtype=bool)
    site_pos = np.stack([f["site_pos"] for f in frames])
    site_mat = np.stack([f["site_mat"] for f in frames])
    for b, f in enumerate(frames):
        n = f["ncon"]
        if n == 0:
            continue
        pos[b, :n] = f["contact_pos"]
        frame[b, :n] = f["contact_frame"]
        force[b, :n] = f["contact_force"]
        sign[b, :n] = f["contact_sign"]
        body[b, :n] = f["contact_body"]
        valid[b, :n] = f["contact_valid"]
    return encode_taxels(
        contact_pos=_t(pos),
        contact_frame=_t(frame),
        contact_force=_t(force),
        contact_sign=_t(sign),
        contact_body=_t(body, torch.int64),
        contact_valid=_t(valid, torch.bool),
        site_pos=_t(site_pos),
        site_mat=_t(site_mat),
        taxel_body=taxel_body,
        kernel_sigma=sigma,
        kernel_cutoff=cutoff,
    )


def _errors(got: np.ndarray, ref: np.ndarray):
    """Max abs and max relative error per channel over the flattened frames."""
    diff = np.abs(got - ref)
    abs_err = diff.reshape(-1, 3).max(axis=0)
    # Relative error is only meaningful where the reference is non-trivial; most
    # of the 368 taxels are exactly zero in every frame.
    ref_mag = np.abs(ref).reshape(-1, 3)
    d = diff.reshape(-1, 3)
    rel_err = np.zeros(3)
    for c in range(3):
        live = ref_mag[:, c] > 1e-6
        rel_err[c] = (d[live, c] / ref_mag[live, c]).max() if live.any() else 0.0
    return abs_err, rel_err


def test_encoder_matches_oracle():
    static, frames, sigma, cutoff = _build_inputs()
    taxel_body = torch.as_tensor(static.taxel_body, dtype=torch.int64)

    per_frame = np.stack([_encode_one(f, taxel_body, sigma, cutoff).numpy() for f in frames])
    ref = np.stack([f["taxels"] for f in frames])
    batched = _encode_batched(frames, taxel_body, sigma, cutoff).numpy()

    assert not np.isnan(per_frame).any(), "NaN in per-frame output"
    assert not np.isnan(batched).any(), "NaN in batched output"

    abs_err, rel_err = _errors(per_frame, ref)
    b_abs, b_rel = _errors(batched, ref)
    batch_delta = np.abs(batched - per_frame).reshape(-1, 3).max(axis=0)

    empty = [i for i, f in enumerate(frames) if f["ncon"] == 0]
    nonempty = [i for i, f in enumerate(frames) if f["ncon"] > 0]

    print(f"\noracle: {len(frames)} frames, {ref.shape[1]} taxels, "
          f"sigma={sigma}, cutoff={cutoff}")
    print(f"  ncon==0 frames: {empty}")
    print(f"  peak |ref| per channel: "
          + "  ".join(f"{c}={np.abs(ref[..., k]).max():.4f}" for k, c in enumerate(CHANNELS)))
    print(f"\n{'channel':<10}{'max abs err':>14}{'max rel err':>14}"
          f"{'batched abs':>14}{'batched rel':>14}{'batch-vs-1':>13}")
    for k, name in enumerate(CHANNELS):
        print(f"{name:<10}{abs_err[k]:>14.3e}{rel_err[k]:>14.3e}"
              f"{b_abs[k]:>14.3e}{b_rel[k]:>14.3e}{batch_delta[k]:>13.3e}")

    # ncon == 0 must give exact zeros, not NaN from a 0/0 normalisation.
    for i in empty:
        assert np.array_equal(per_frame[i], np.zeros_like(per_frame[i])), \
            f"frame {i} has ncon=0 but non-zero output"
        assert np.array_equal(batched[i], np.zeros_like(batched[i])), \
            f"frame {i} (batched) has ncon=0 but non-zero output"
        assert np.array_equal(ref[i], np.zeros_like(ref[i])), \
            f"frame {i} oracle is not zero — fixture assumption broken"

    # Sanity: the fixture must actually exercise the encoder, or this test is
    # only proving that zero equals zero.
    assert nonempty, "fixture has no contact frames"
    assert np.abs(ref[nonempty]).max() > 1e-3, "fixture contains no real forces"

    assert abs_err.max() < ABS_TOL, f"per-frame abs error {abs_err} exceeds {ABS_TOL}"
    assert b_abs.max() < ABS_TOL, f"batched abs error {b_abs} exceeds {ABS_TOL}"
    assert batch_delta.max() < BATCH_TOL, \
        f"batched result differs from per-frame by {batch_delta}"


if __name__ == "__main__":
    test_encoder_matches_oracle()
    print("\nOK")
