"""Validate the Triton taxel encoder against the oracle AND the torch reference.

WHY two references and not one: `src/kernels/triton/taxel_triton.py` is a
reformulation, not a translation. It replaces a dense `(B, C, T, .)` evaluation
with a two-pass gather that recomputes weights, skips padded contact slots, and
skips taxel tiles belonging to other bodies. Every one of those skips is an
opportunity to produce a plausible-looking tactile map that is quietly wrong —
which is the failure mode this whole line of work guards against
(PROJECT_LOG 1.5b/1.5d, and `benchmarks/harness.py`'s "a fast wrong number is
worth nothing").

So:
  1. against `explore/out/oracle_fixture.npz` — the banked CPU reference encoder's
     inputs and outputs, the same fixture `tests/test_encoder_oracle.py` uses.
     This checks the *semantics* (force sign, cutoffs, patch gate, body masking,
     the final z flip) against something that never went through torch at all.
  2. against `encoders.taxel_torch.encode_taxels` on synthetic batches at
     B = 1, 32, 1024, using `profiling/profile_encoder.py`'s `make_inputs` so the
     input statistics match the ones every profiling number was taken on.

Explicitly covered edge cases, because each has its own way of going wrong:
  * `ncon == 0` (C == 0)     -> zeros, not NaN and not a launch of an empty grid
  * a contact whose weight_sum is 0 (contact far from every taxel of its body,
    or gated out by the patch-locality test) -> that contact contributes nothing
    and the output is 0, not NaN from a 0/0 normalisation
  * padding slots (`contact_valid` false, `contact_body == -1`) -> ignored, and
    ignored even when their `contact_pos`/`contact_force` hold live-looking junk
  * a contact on a body with no taxels at all

Plus a determinism check: 20 runs on identical input, asserted **bit-identical**.
`profiling/RESULTS.md` records that H6's determinism claim was never measured
because the existing path is already deterministic (it sorts rather than using
atomics). This measures it for the new kernel.

Run (memory-capped, as everything on this box must be — unified memory means an
oversized allocation takes down the host, not the process):

    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
      .venv/bin/python tests/test_triton_kernel.py
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _extra in (ROOT / "src", ROOT / "explore", ROOT / "third_party" / "leapXELA_model"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from encoders.taxel_torch import (  # noqa: E402
    N_TAXELS,
    contact_inputs_from_geoms,
    encode_taxels,
    prepare_static,
)
from kernels.triton.taxel_triton import encode_taxels_triton  # noqa: E402

FIXTURE = ROOT / "explore" / "out" / "oracle_fixture.npz"
CHANNELS = ("shear_x", "shear_y", "normal_z")
ABS_TOL = 1e-4      # the mandated gate; achieved error is reported alongside

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_inputs(B: int, seed: int = 0):
    """Reuse the profiler's synthetic generator rather than reimplementing it.

    Every number in `profiling/RESULTS.md` was taken on this distribution
    (~4.3 live contacts of 48 slots, contacts within ~sigma of a real taxel,
    genuine rotation matrices). A local reimplementation that drifted would make
    the correctness check and the benchmark disagree about what "the workload" is.
    """
    pe = _load_by_path("profile_encoder_ref", ROOT / "profiling" / "profile_encoder.py")
    return pe.make_inputs(B, seed=seed)


def _errors(got: np.ndarray, ref: np.ndarray):
    """Max abs and max relative error per channel. Relative error is only taken
    where the reference is non-trivial — most of the 368 taxels are exactly zero
    in every frame, and 0/0 would report nothing useful."""
    diff = np.abs(got - ref).reshape(-1, 3)
    ref_mag = np.abs(ref).reshape(-1, 3)
    abs_err = diff.max(axis=0)
    rel_err = np.zeros(3)
    for c in range(3):
        live = ref_mag[:, c] > 1e-6
        rel_err[c] = (diff[live, c] / ref_mag[live, c]).max() if live.any() else 0.0
    return abs_err, rel_err


def _report(title: str, abs_err, rel_err):
    print(f"\n{title}")
    print(f"{'channel':<10}{'max abs err':>15}{'max rel err':>15}")
    for k, name in enumerate(CHANNELS):
        print(f"{name:<10}{abs_err[k]:>15.3e}{rel_err[k]:>15.3e}")


# --------------------------------------------------------------------------- #
# 1. Oracle fixture
# --------------------------------------------------------------------------- #


def _oracle_frames():
    """Fixture + model -> per-frame numpy inputs, exactly as the oracle test does."""
    from leapxela.taxel_layout import build_layout
    from taxel_map import remap_layout

    ref = _load_by_path("run_encoder_ref", ROOT / "explore" / "04_run_encoder.py")
    model = ref.build_model()
    layout = remap_layout(build_layout())
    static = prepare_static(model, layout, ref.cube_geom_names(model))

    npz = np.load(FIXTURE)
    n_frames = int(npz["n_frames"])
    frames = []
    for i in range(n_frames):
        p = f"f{i:03d}_"
        ncon = int(npz[p + "ncon"])
        body, sign, valid = contact_inputs_from_geoms(
            static, npz[p + "contact_geom1"].astype(np.int64),
            npz[p + "contact_geom2"].astype(np.int64),
        )
        frames.append(dict(
            ncon=ncon,
            contact_pos=npz[p + "contact_pos"].reshape(ncon, 3),
            contact_frame=npz[p + "contact_frame"].reshape(ncon, 3, 3),
            contact_force=npz[p + "contact_force"].reshape(ncon, 6),
            contact_body=body, contact_sign=sign, contact_valid=valid,
            site_pos=npz[p + "site_xpos"][static.site_ids],
            site_mat=npz[p + "site_xmat"][static.site_ids].reshape(-1, 3, 3),
            taxels=npz[p + "taxels"],
        ))
    return static, frames, float(npz["kernel_sigma"]), float(npz["kernel_cutoff"])


def _pad_batch(frames, static, dev):
    """All frames stacked into one padded (B, C, ...) CUDA batch, C = max ncon.

    Padding is the point: slots beyond a frame's own ncon are exactly the
    `contact_valid == False` / `contact_body == -1` case the kernel must ignore.
    """
    B = len(frames)
    C = max(1, max(f["ncon"] for f in frames))
    T = N_TAXELS
    pos = np.zeros((B, C, 3), np.float32)
    frame = np.zeros((B, C, 3, 3), np.float32)
    force = np.zeros((B, C, 6), np.float32)
    sign = np.zeros((B, C), np.float32)
    body = np.full((B, C), -1, np.int64)
    valid = np.zeros((B, C), bool)
    for b, f in enumerate(frames):
        n = f["ncon"]
        if n == 0:
            continue
        pos[b, :n], frame[b, :n], force[b, :n] = (
            f["contact_pos"], f["contact_frame"], f["contact_force"])
        sign[b, :n], body[b, :n], valid[b, :n] = (
            f["contact_sign"], f["contact_body"], f["contact_valid"])
    t = lambda x, d=torch.float32: torch.as_tensor(  # noqa: E731
        np.ascontiguousarray(x), dtype=d, device=dev)
    return dict(
        contact_pos=t(pos), contact_frame=t(frame), contact_force=t(force),
        contact_sign=t(sign), contact_body=t(body, torch.int64),
        contact_valid=t(valid, torch.bool),
        site_pos=t(np.stack([f["site_pos"] for f in frames])),
        site_mat=t(np.stack([f["site_mat"] for f in frames])),
        taxel_body=t(static.taxel_body, torch.int64),
    ), B, C, T


def test_triton_matches_oracle():
    """Semantics, against the banked CPU reference encoder's own output."""
    dev = torch.device("cuda")
    static, frames, sigma, cutoff = _oracle_frames()
    batch, B, C, T = _pad_batch(frames, static, dev)

    got = encode_taxels_triton(
        **batch, kernel_sigma=sigma, kernel_cutoff=cutoff).cpu().numpy()
    ref = np.stack([f["taxels"] for f in frames])

    assert not np.isnan(got).any(), "NaN in Triton output"
    abs_err, rel_err = _errors(got, ref)

    empty = [i for i, f in enumerate(frames) if f["ncon"] == 0]
    nonempty = [i for i, f in enumerate(frames) if f["ncon"] > 0]
    print(f"\noracle: {B} frames padded to C={C}, T={T}, "
          f"sigma={sigma}, cutoff={cutoff}; ncon==0 frames {empty}")
    _report("Triton vs oracle fixture", abs_err, rel_err)

    # The fixture must actually exercise the encoder, or this proves 0 == 0.
    assert nonempty, "fixture has no contact frames"
    assert np.abs(ref[nonempty]).max() > 1e-3, "fixture contains no real forces"

    # PADDING SLOTS: every frame with ncon < C carries them, and the ncon == 0
    # frames are nothing but padding. Those must give exact zeros.
    for i in empty:
        assert np.array_equal(got[i], np.zeros_like(got[i])), \
            f"frame {i} has ncon=0 but the Triton output is non-zero"

    assert abs_err.max() < ABS_TOL, f"abs error {abs_err} exceeds {ABS_TOL}"


# --------------------------------------------------------------------------- #
# 2. Torch reference, several B
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("B", [1, 32, 1024])
def test_triton_matches_torch(B: int):
    """Same maths, at the batch sizes the benchmark quotes."""
    inputs = _make_inputs(B)
    ref = encode_taxels(**inputs)
    got = encode_taxels_triton(**inputs)

    assert not torch.isnan(got).any(), f"NaN in Triton output at B={B}"
    abs_err, rel_err = _errors(got.cpu().numpy(), ref.cpu().numpy())
    _report(f"Triton vs torch reference, B={B} "
            f"(peak |ref| = {ref.abs().max().item():.4f})", abs_err, rel_err)
    assert abs_err.max() < ABS_TOL, f"B={B}: abs error {abs_err} exceeds {ABS_TOL}"


# --------------------------------------------------------------------------- #
# 3. Edge cases
# --------------------------------------------------------------------------- #


def _tiny_case(C: int, sigma: float = 0.0035, cutoff: float = 0.01, T: int = 8,
               B: int = 3, dev="cuda"):
    """A hand-built batch small enough to reason about slot by slot."""
    g = torch.Generator(device=dev).manual_seed(7)
    taxel_body = torch.arange(T, device=dev) // 2          # 4 bodies, 2 taxels each
    site_pos = torch.rand((B, T, 3), generator=g, device=dev) * 0.05
    # Identity site frames keep the local/world distinction out of the way; the
    # rotation maths is already covered by the oracle and torch comparisons.
    site_mat = torch.eye(3, device=dev).expand(B, T, 3, 3).contiguous()
    return dict(
        contact_pos=torch.zeros((B, C, 3), device=dev),
        contact_frame=torch.eye(3, device=dev).expand(B, C, 3, 3).contiguous(),
        contact_force=torch.zeros((B, C, 6), device=dev),
        contact_sign=torch.ones((B, C), device=dev),
        contact_body=torch.full((B, C), -1, dtype=torch.int64, device=dev),
        contact_valid=torch.zeros((B, C), dtype=torch.bool, device=dev),
        site_pos=site_pos, site_mat=site_mat, taxel_body=taxel_body,
        kernel_sigma=sigma, kernel_cutoff=cutoff,
    )


def test_ncon_zero():
    """C == 0: zeros of the right shape, not NaN and not a zero-sized launch."""
    inp = _tiny_case(C=0)
    got = encode_taxels_triton(**inp)
    ref = encode_taxels(**inp)
    assert got.shape == (3, 8, 3)
    assert torch.equal(got, torch.zeros_like(got)), "ncon=0 gave non-zero output"
    assert torch.equal(got, ref)


def test_weight_sum_zero():
    """A live contact whose weights all vanish must emit zeros, not NaN.

    Two independent ways to get weight_sum == 0, and both are exercised:
      slot 0 — in-plane distance beyond `kernel_cutoff` from every taxel;
      slot 1 — in-plane distance ~0 but out of plane by more than the cutoff, so
               the patch-locality gate zeroes it (the reference's `|local_z| >
               cutoff`). This is the one that would silently pass if the gate
               were dropped, because the Gaussian itself would be ~1.
    """
    inp = _tiny_case(C=4)
    dev = inp["site_pos"].device
    for b in range(3):
        # slot 0: far away in-plane, on body 0
        inp["contact_pos"][b, 0] = inp["site_pos"][b, 0] + torch.tensor(
            [1.0, 0.0, 0.0], device=dev)
        inp["contact_body"][b, 0] = 0
        inp["contact_valid"][b, 0] = True
        inp["contact_force"][b, 0, :3] = torch.tensor([1.0, 2.0, 3.0], device=dev)

        # slot 1: dead-on in-plane but 5x cutoff out of plane, on body 1
        inp["contact_pos"][b, 1] = inp["site_pos"][b, 2] + torch.tensor(
            [0.0, 0.0, 5 * inp["kernel_cutoff"]], device=dev)
        inp["contact_body"][b, 1] = 1
        inp["contact_valid"][b, 1] = True
        inp["contact_force"][b, 1, :3] = torch.tensor([4.0, 5.0, 6.0], device=dev)

        # slot 2: valid, forceful, but on a body that carries no taxels at all
        inp["contact_pos"][b, 2] = inp["site_pos"][b, 4]
        inp["contact_body"][b, 2] = 99
        inp["contact_valid"][b, 2] = True
        inp["contact_force"][b, 2, :3] = torch.tensor([7.0, 8.0, 9.0], device=dev)

        # slot 3: PADDING carrying live-looking junk. contact_valid is False and
        # contact_body is -1; if either is ignored this slot would dominate.
        inp["contact_pos"][b, 3] = inp["site_pos"][b, 6]
        inp["contact_force"][b, 3, :3] = torch.tensor([1e3, 1e3, 1e3], device=dev)

    got = encode_taxels_triton(**inp)
    ref = encode_taxels(**inp)
    assert not torch.isnan(got).any(), "NaN from a zero weight_sum"
    assert torch.equal(got, torch.zeros_like(got)), \
        f"zero-weight contacts leaked force: max |out| = {got.abs().max().item():.3e}"
    assert torch.allclose(got, ref, atol=1e-7)
    print("\nweight_sum==0 / no-taxel body / junk padding: all exactly zero, no NaN")


def test_padding_slots_ignored():
    """Padding must be ignored even when it duplicates a live contact.

    Slot 0 is live; slots 1..C-1 are byte-identical copies of it with
    `contact_valid` cleared. If padding were being read, the output would be a
    factor of C too large — and would still look like a plausible tactile map.
    """
    inp = _tiny_case(C=16)
    dev = inp["site_pos"].device
    for b in range(3):
        inp["contact_pos"][b, :] = inp["site_pos"][b, 0]
        inp["contact_force"][b, :, :3] = torch.tensor([0.5, -0.25, 2.0], device=dev)
        inp["contact_body"][b, :] = 0
        inp["contact_valid"][b, 0] = True     # only slot 0 is live
    inp["contact_body"][:, 1:] = -1

    got = encode_taxels_triton(**inp)
    ref = encode_taxels(**inp)
    assert got.abs().max() > 1e-3, "test is vacuous: the live contact produced nothing"
    err = (got - ref).abs().max().item()
    print(f"\npadding slots (15 dead copies of 1 live contact): max abs err {err:.3e}, "
          f"peak |out| {got.abs().max().item():.4f}")
    assert err < 1e-6, f"padding leaked: {err}"


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #


def test_determinism():
    """20 runs on identical input, bit-identical output.

    This is the claim H6 made and `profiling/RESULTS.md` recorded as unmeasured.
    The kernel accumulates in registers in a fixed order (ascending contact slot)
    with no atomics, so bit-identity is expected by construction — but "expected
    by construction" is what this repo keeps finding to be false, so measure it.
    """
    inputs = _make_inputs(1024)
    first = encode_taxels_triton(**inputs).clone()
    for i in range(1, 20):
        again = encode_taxels_triton(**inputs)
        assert torch.equal(first, again), (
            f"run {i} differs from run 0: max abs delta "
            f"{(again - first).abs().max().item():.3e}")
    print(f"\ndeterminism: 20/20 runs bit-identical at B=1024 "
          f"({first.numel():,} floats, peak |out| {first.abs().max().item():.4f})")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_triton_matches_oracle()
    for _b in (1, 32, 1024):
        test_triton_matches_torch(_b)
    test_ncon_zero()
    test_weight_sum_zero()
    test_padding_slots_ignored()
    test_determinism()
    print("\nOK")
