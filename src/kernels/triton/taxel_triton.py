"""Fused Triton implementation of the Gaussian-splat taxel encoder.

WHY THIS EXISTS
---------------
`src/encoders/taxel_torch.py` is the readable dense baseline. `profiling/RESULTS.md`
established, by measurement, exactly what is wrong with it — and the design below
is dictated by that verdict, not chosen for elegance:

  * The encoder is **bandwidth-bound**: arithmetic intensity 0.192 FLOP/byte
    against a *measured* ridge point of 70.7 for this GB10 (368x to the left).
    It already runs at 71% of the measured 260.8 GB/s DRAM ceiling and 0.19% of
    the SGEMM ceiling. There is nothing left to win inside those kernels.
  * The cost is **materialising `(B, C, T, .)` intermediates** — 868 MB each at
    N=4096, 19.99 GB of DRAM traffic per call — of which **99.3% is multiplied
    by zero** after having been written out and read back (measured 3.92 live
    contacts against a 48-slot budget, then a body mask that keeps 1/12 of the
    taxels: 72.4M triples computed, ~492k carry signal).
  * **Launch overhead is 0.18% of the call** (58 launches x 3.34 us). So this
    file does NOT try to be one kernel. Two kernels is fine, and the two-pass
    structure below is worth far more than the launch it costs.
  * The compulsory traffic floor — inputs plus output, nothing else — is
    **103.8 MB at N=4096**, i.e. 193x less than today.
  * **`torch.compile` on the unmodified encoder already gives 2.41x**, so that,
    not eager, is the baseline this has to beat.

THE DESIGN
----------
Two-pass gather, recomputing the weights in pass 2 rather than storing them.

The two-pass split is *forced*, not preferred: `weights /= weights.sum()`
reduces over the taxel axis, and the taxel axis is pass 2's parallel axis, so
pass 2 cannot compute its own denominator. Recompute beats store because traffic
costs ~368x what FLOPs cost here; the only thing that reaches DRAM between the
passes is one float per (env, contact) — 786 KB at N=4096.

  Pass 1 (`_weight_sum_kernel`) — one program per (env, contact). Walks the
  taxels, computes each Gaussian weight exactly as the reference does, and
  writes a single float: `weight_sum[B, C]`. No collisions, no atomics.

  Pass 2 (`_gather_kernel`) — one program per (env, taxel tile). Loads that
  tile's site pose once, loops the env's contact slots, recomputes the weight,
  divides by `weight_sum`, rotates the contact wrench into the taxel frame and
  accumulates **in registers**. Three stores at the end.

Nothing of shape `(B, C, T, .)` is ever created. That is the entire point.

Two data-dependent skips, both justified by the measured sparsity rather than by
taste:
  * a contact slot whose `contact_valid` is false, or whose `weight_sum` is <= 0,
    is skipped with a *uniform* branch after three scalar loads. ~44 of 48 slots
    per env are padding (~11x).
  * a taxel tile that contains no taxel of the contact's body is skipped
    entirely. The body mask keeps ~31 of 368 taxels (~12x). Note this skip is an
    arithmetic saving only — the masked loads it avoids were already generating
    no DRAM traffic, because a predicated-off load does not touch memory.

DETERMINISM
-----------
Both passes accumulate in registers in a fixed order (taxel-tile order in pass 1,
ascending contact slot in pass 2), and neither uses atomics, so the output is
bit-identical run to run. `profiling/RESULTS.md` notes that H6's determinism
claim was never actually measured; `tests/test_triton_kernel.py` measures it.

NUMERICS
--------
The arithmetic mirrors `encode_taxels` operation for operation and in the same
association order (including `-dist_sq / (2*sigma^2)` as negate-then-divide, and
the `[1, 1, -1]` flip applied last), so the two agree to float32 rounding. They
are not bit-identical: the weight sum and the contact accumulation are summed in
a different order, and float addition is not associative. Measured agreement is
reported by the test, not asserted here.

MEASURED (2026-08-13, same box and same synthetic inputs as the profiling run;
full sweep in `benchmarks/results/kernel_bench.csv`)
-----------------------------------------------------------------------------
At N=4096: **0.582 ms/call** against 109.5 ms eager (**188x**) and 45.5 ms
`torch.compile` (**78x**), max abs error 1.8e-06, output bit-identical over 20
runs. Modelled traffic 136 MB against the 103.8 MB floor (1.31x the floor,
193x less than eager's 19.99 GB); 233.8 GB/s, 90% of the measured 260.8 GB/s
ceiling. Peak torch allocation drops 4.095 GB -> 0.183 GB.

Three caveats, because this beats even the optimistic bound in RESULTS.md:
  * achieved GB/s is **modelled bytes / measured time**, not a counter read —
    `ncu` cannot read a counter on this box (`ERR_NVGPUCTRPERM`). At N=512-2048
    the figure lands at 95-99% of the DRAM ceiling, which almost certainly means
    L2 reuse is being charged as DRAM traffic, not that the wall was beaten.
  * unlike the dense baseline, **this kernel's runtime is data-dependent**, so
    the profiling report's "synthetic == real to 1.0006" result does NOT carry
    over. Measured sensitivity at N=4096: 0.54 ms at 1 live contact/env, 0.57 at
    the real 4.3, 0.73 at 12, and 1.27 ms with all 48 slots live — so even the
    fully saturated worst case is 86x eager / 36x `torch.compile`.
  * Amdahl still binds the project: the encoder was 77.1% of step time, so total
    step speedup caps at 4.37x no matter how fast this gets. Past ~30x the
    honest next move is to profile the physics, not this file.

SAFETY: this box has ~121 GB of *unified* CPU/GPU memory, so an oversized
allocation wedges the host rather than the process. Run everything under
    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 ...
and do not exceed 4096 environments.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Kept in sync with the reference module rather than re-derived.
N_TAXELS = 368


# --------------------------------------------------------------------------- #
# Pass 1: per-contact weight normaliser
# --------------------------------------------------------------------------- #


@triton.jit
def _weight_sum_kernel(
    contact_pos_ptr,      # (B, C, 3)  f32
    contact_body_ptr,     # (B, C)     int
    contact_valid_ptr,    # (B, C)     int8 (bool viewed as int8)
    site_pos_ptr,         # (B, T, 3)  f32
    site_mat_ptr,         # (B, T, 3, 3) f32
    taxel_body_ptr,       # (T,)       int
    weight_sum_ptr,       # (B, C)     f32   <- the only thing written
    C,
    T,
    two_sigma_sq,         # 2 * kernel_sigma**2
    cutoff,               # kernel_cutoff
    cutoff_sq,            # kernel_cutoff**2
    BLOCK_T: tl.constexpr,
):
    """One program per (env, contact): sum this contact's splat over all taxels.

    Reduces over the taxel axis, which is exactly the axis pass 2 parallelises
    over — hence the separate pass. Writes one float and reads only the taxels of
    the body this contact touched.
    """
    pid = tl.program_id(0)
    b = pid // C

    body = tl.load(contact_body_ptr + pid)
    valid = tl.load(contact_valid_ptr + pid) != 0

    total = 0.0
    # Purely a performance skip: a padded slot has body == -1, matches no taxel,
    # and would sum to 0.0 anyway. Correctness does not depend on this branch.
    if valid & (body >= 0):
        cx = tl.load(contact_pos_ptr + pid * 3 + 0)
        cy = tl.load(contact_pos_ptr + pid * 3 + 1)
        cz = tl.load(contact_pos_ptr + pid * 3 + 2)

        pos_base = b * T * 3
        mat_base = b * T * 9

        for t0 in range(0, T, BLOCK_T):
            offs = t0 + tl.arange(0, BLOCK_T)
            tmask = offs < T
            tb = tl.load(taxel_body_ptr + offs, mask=tmask, other=-1)
            m = tmask & (tb == body)

            # Skip tiles belonging to other bodies. The masked loads below would
            # already move no DRAM; this skips the arithmetic too (~12x).
            if tl.max(m.to(tl.int32)) > 0:
                sp = pos_base + offs * 3
                sx = tl.load(site_pos_ptr + sp + 0, mask=m, other=0.0)
                sy = tl.load(site_pos_ptr + sp + 1, mask=m, other=0.0)
                sz = tl.load(site_pos_ptr + sp + 2, mask=m, other=0.0)

                sm = mat_base + offs * 9
                r00 = tl.load(site_mat_ptr + sm + 0, mask=m, other=0.0)
                r01 = tl.load(site_mat_ptr + sm + 1, mask=m, other=0.0)
                r02 = tl.load(site_mat_ptr + sm + 2, mask=m, other=0.0)
                r10 = tl.load(site_mat_ptr + sm + 3, mask=m, other=0.0)
                r11 = tl.load(site_mat_ptr + sm + 4, mask=m, other=0.0)
                r12 = tl.load(site_mat_ptr + sm + 5, mask=m, other=0.0)
                r20 = tl.load(site_mat_ptr + sm + 6, mask=m, other=0.0)
                r21 = tl.load(site_mat_ptr + sm + 7, mask=m, other=0.0)
                r22 = tl.load(site_mat_ptr + sm + 8, mask=m, other=0.0)

                # rel = contact_pos - site_pos, then local = R^T rel, i.e.
                # local_i = sum_j site_mat[..., j, i] * rel_j  (site_mat is
                # local->world, so its transpose maps world->local).
                rx = cx - sx
                ry = cy - sy
                rz = cz - sz
                lx = r00 * rx + r10 * ry + r20 * rz
                ly = r01 * rx + r11 * ry + r21 * rz
                lz = r02 * rx + r12 * ry + r22 * rz

                # In-plane distance only; the out-of-plane component is a gate,
                # not part of the Gaussian.
                d2 = lx * lx + ly * ly
                w = tl.exp(-d2 / two_sigma_sq)
                w = tl.where(d2 > cutoff_sq, 0.0, w)
                # Patch-locality gate: taxels whose plane is far from the contact
                # belong to a different patch of the same body (PROJECT_LOG 1.5c).
                w = tl.where(tl.abs(lz) > cutoff, 0.0, w)
                w = tl.where(m, w, 0.0)

                total += tl.sum(w, axis=0)

    tl.store(weight_sum_ptr + pid, total)


# --------------------------------------------------------------------------- #
# Pass 2: per-taxel gather
# --------------------------------------------------------------------------- #


@triton.jit
def _gather_kernel(
    contact_pos_ptr,      # (B, C, 3)
    contact_frame_ptr,    # (B, C, 3, 3)
    contact_force_ptr,    # (B, C, 6)   only [:3] is read
    contact_sign_ptr,     # (B, C)
    contact_body_ptr,     # (B, C)
    contact_valid_ptr,    # (B, C) int8
    weight_sum_ptr,       # (B, C)      from pass 1
    site_pos_ptr,         # (B, T, 3)
    site_mat_ptr,         # (B, T, 3, 3)
    taxel_body_ptr,       # (T,)
    out_ptr,              # (B, T, 3)
    C,
    T,
    two_sigma_sq,
    cutoff,
    cutoff_sq,
    BLOCK_T: tl.constexpr,
):
    """One program per (env, taxel tile): accumulate every contact into registers.

    The site pose is loaded once and reused across every contact slot, so the
    dominant read (`site_pos` + `site_mat`, 72.3 MB at N=4096) is touched exactly
    once per call. Nothing of shape (B, C, T, .) is formed.
    """
    b = tl.program_id(0)
    t0 = tl.program_id(1) * BLOCK_T
    offs = t0 + tl.arange(0, BLOCK_T)
    tmask = offs < T

    tb = tl.load(taxel_body_ptr + offs, mask=tmask, other=-1)

    sp = b * T * 3 + offs * 3
    sx = tl.load(site_pos_ptr + sp + 0, mask=tmask, other=0.0)
    sy = tl.load(site_pos_ptr + sp + 1, mask=tmask, other=0.0)
    sz = tl.load(site_pos_ptr + sp + 2, mask=tmask, other=0.0)

    sm = b * T * 9 + offs * 9
    r00 = tl.load(site_mat_ptr + sm + 0, mask=tmask, other=0.0)
    r01 = tl.load(site_mat_ptr + sm + 1, mask=tmask, other=0.0)
    r02 = tl.load(site_mat_ptr + sm + 2, mask=tmask, other=0.0)
    r10 = tl.load(site_mat_ptr + sm + 3, mask=tmask, other=0.0)
    r11 = tl.load(site_mat_ptr + sm + 4, mask=tmask, other=0.0)
    r12 = tl.load(site_mat_ptr + sm + 5, mask=tmask, other=0.0)
    r20 = tl.load(site_mat_ptr + sm + 6, mask=tmask, other=0.0)
    r21 = tl.load(site_mat_ptr + sm + 7, mask=tmask, other=0.0)
    r22 = tl.load(site_mat_ptr + sm + 8, mask=tmask, other=0.0)

    ax = tl.zeros([BLOCK_T], dtype=tl.float32)
    ay = tl.zeros([BLOCK_T], dtype=tl.float32)
    az = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Ascending slot order, fixed and register-local -> bit-reproducible.
    for c in range(0, C):
        idx = b * C + c
        body = tl.load(contact_body_ptr + idx)
        valid = tl.load(contact_valid_ptr + idx) != 0
        wsum = tl.load(weight_sum_ptr + idx)
        m = tmask & (tb == body)

        # `wsum <= 0` reproduces the reference's "skip this contact": all weights
        # are >= 0, so a zero sum means every weight was already zero. Skipping is
        # exact, and it is also what keeps 0/0 from producing NaN.
        if valid & (wsum > 0.0) & (tl.max(m.to(tl.int32)) > 0):
            cx = tl.load(contact_pos_ptr + idx * 3 + 0)
            cy = tl.load(contact_pos_ptr + idx * 3 + 1)
            cz = tl.load(contact_pos_ptr + idx * 3 + 2)

            rx = cx - sx
            ry = cy - sy
            rz = cz - sz
            lx = r00 * rx + r10 * ry + r20 * rz
            ly = r01 * rx + r11 * ry + r21 * rz
            lz = r02 * rx + r12 * ry + r22 * rz

            d2 = lx * lx + ly * ly
            w = tl.exp(-d2 / two_sigma_sq)
            w = tl.where(d2 > cutoff_sq, 0.0, w)
            w = tl.where(tl.abs(lz) > cutoff, 0.0, w)
            w = tl.where(m, w, 0.0)
            w = w / wsum

            # Contact wrench -> world: force_world_i = sum_j frame[j, i] * f_j,
            # then the object/hand sign flip. `contact_force` is mj_contactForce
            # output in the contact frame; only the linear part is used.
            f0 = tl.load(contact_force_ptr + idx * 6 + 0)
            f1 = tl.load(contact_force_ptr + idx * 6 + 1)
            f2 = tl.load(contact_force_ptr + idx * 6 + 2)
            k00 = tl.load(contact_frame_ptr + idx * 9 + 0)
            k01 = tl.load(contact_frame_ptr + idx * 9 + 1)
            k02 = tl.load(contact_frame_ptr + idx * 9 + 2)
            k10 = tl.load(contact_frame_ptr + idx * 9 + 3)
            k11 = tl.load(contact_frame_ptr + idx * 9 + 4)
            k12 = tl.load(contact_frame_ptr + idx * 9 + 5)
            k20 = tl.load(contact_frame_ptr + idx * 9 + 6)
            k21 = tl.load(contact_frame_ptr + idx * 9 + 7)
            k22 = tl.load(contact_frame_ptr + idx * 9 + 8)
            sgn = tl.load(contact_sign_ptr + idx)

            fwx = (k00 * f0 + k10 * f1 + k20 * f2) * sgn
            fwy = (k01 * f0 + k11 * f1 + k21 * f2) * sgn
            fwz = (k02 * f0 + k12 * f1 + k22 * f2) * sgn

            # ...and world -> taxel local, same transpose as `local` above.
            flx = r00 * fwx + r10 * fwy + r20 * fwz
            fly = r01 * fwx + r11 * fwy + r21 * fwz
            flz = r02 * fwx + r12 * fwy + r22 * fwz

            ax += w * flx
            ay += w * fly
            az += w * flz

    ob = b * T * 3 + offs * 3
    tl.store(out_ptr + ob + 0, ax, mask=tmask)
    tl.store(out_ptr + ob + 1, ay, mask=tmask)
    # `out[:, 2] *= -1` from the reference: compression positive. Negation is
    # exact, so applying it here rather than in a separate pass changes nothing.
    tl.store(out_ptr + ob + 2, -az, mask=tmask)


# --------------------------------------------------------------------------- #
# Host wrapper
# --------------------------------------------------------------------------- #

# Chosen by measurement, not by taste: a 3x3 grid sweep at B = 128 / 1024 / 4096.
# Pass 2 is strongly tile-sensitive (0.57 / 0.67 / 0.87 ms at BLOCK_T = 32 / 64 /
# 16, B=4096) — 32 wins because it gives 12 programs per env, enough to fill 48
# SMs, without spilling the ~40 live registers each program holds. Pass 1 barely
# cares (0.57-0.64 ms across 32/64/128); 64 is best by ~1%, which is inside the
# run-to-run spread, so do not read much into it.
BLOCK_T_PASS1 = 64
BLOCK_T_PASS2 = 32


def _num_warps(block: int) -> int:
    """One warp per 32 lanes. Triton's default of 4 leaves most threads idle for
    the small tiles used here, and idle threads still cost occupancy."""
    return max(1, min(8, block // 32))


def encode_taxels_triton(
    contact_pos: torch.Tensor,     # (B, C, 3)
    contact_frame: torch.Tensor,   # (B, C, 3, 3)
    contact_force: torch.Tensor,   # (B, C, 6)
    contact_sign: torch.Tensor,    # (B, C)
    contact_body: torch.Tensor,    # (B, C)   body slot, -1 = invalid/padding
    contact_valid: torch.Tensor,   # (B, C)   bool
    site_pos: torch.Tensor,        # (B, T, 3)
    site_mat: torch.Tensor,        # (B, T, 3, 3)
    taxel_body: torch.Tensor,      # (T,)
    kernel_sigma: float,
    kernel_cutoff: float,
    block_t_pass1: int = BLOCK_T_PASS1,
    block_t_pass2: int = BLOCK_T_PASS2,
) -> torch.Tensor:
    """Drop-in replacement for `encoders.taxel_torch.encode_taxels`.

    Same signature, same semantics, same (B, T, 3) output in taxel-local frames
    with compression positive. CUDA + float32 only; everything else raises rather
    than silently falling back, because a silent fallback would make a benchmark
    of this file meaningless.
    """
    B, C = contact_pos.shape[0], contact_pos.shape[1]
    T = site_pos.shape[1]
    dtype, device = contact_pos.dtype, contact_pos.device

    if C == 0:  # ncon == 0 everywhere: zeros, not NaN. Matches the reference.
        return torch.zeros((B, T, 3), dtype=dtype, device=device)

    if device.type != "cuda":
        raise RuntimeError("encode_taxels_triton requires CUDA tensors")
    if dtype != torch.float32:
        raise TypeError(f"encode_taxels_triton is float32-only, got {dtype}")

    # The kernels index with shape-derived strides, so every input must be
    # contiguous. On the harness path they already are (the contact buffers are
    # `.view()`s of contiguous scratch, the site tensors come from an advanced
    # index), so these are no-ops there.
    contact_pos = contact_pos.contiguous()
    contact_frame = contact_frame.contiguous()
    contact_force = contact_force.contiguous()
    contact_sign = contact_sign.contiguous().to(dtype)
    contact_body = contact_body.contiguous()
    site_pos = site_pos.contiguous()
    site_mat = site_mat.contiguous()
    taxel_body = taxel_body.contiguous()

    # bool -> int8 is a free reinterpret (both are 1 byte); Triton's handling of
    # i1 pointers is not something to rely on.
    valid_i8 = contact_valid.contiguous().view(torch.int8)

    two_sigma_sq = float(2.0 * kernel_sigma**2)
    cutoff = float(kernel_cutoff)
    cutoff_sq = float(kernel_cutoff**2)

    weight_sum = torch.empty((B, C), dtype=torch.float32, device=device)
    out = torch.empty((B, T, 3), dtype=torch.float32, device=device)

    _weight_sum_kernel[(B * C,)](
        contact_pos, contact_body, valid_i8,
        site_pos, site_mat, taxel_body, weight_sum,
        C, T, two_sigma_sq, cutoff, cutoff_sq,
        BLOCK_T=block_t_pass1,
        num_warps=_num_warps(block_t_pass1),
    )

    _gather_kernel[(B, triton.cdiv(T, block_t_pass2))](
        contact_pos, contact_frame, contact_force, contact_sign,
        contact_body, valid_i8, weight_sum,
        site_pos, site_mat, taxel_body, out,
        C, T, two_sigma_sq, cutoff, cutoff_sq,
        BLOCK_T=block_t_pass2,
        num_warps=_num_warps(block_t_pass2),
    )
    return out


def traffic_bytes(B: int, C: int, T: int, live_per_env: float = 4.3,
                  bodies: int = 12) -> dict:
    """Modelled compulsory DRAM traffic for this two-pass formulation.

    Stated as a model, not a measurement: `ncu` cannot read a counter on this box
    (`ERR_NVGPUCTRPERM`, profiling/RESULTS.md 0), so achieved GB/s below is
    `modelled bytes / measured time`, exactly as in the profiling report. Compare
    against the 103.8 MB inputs+output floor for N=4096 quoted there.
    """
    f4 = 4
    per_taxel_site = (3 + 9) * f4          # site_pos + site_mat
    # Pass 1: contact metadata for every slot, site pose for the touched body only.
    p1 = B * C * (3 + 1 + 1) * f4 + int(B * live_per_env) * (T / bodies) * per_taxel_site
    p1 += B * C * f4                       # weight_sum written
    # Pass 2: the full site pose once, contact metadata re-read per taxel tile
    # (L2-resident in practice, counted anyway), and the output.
    p2 = B * T * per_taxel_site + B * C * f4 + B * C * (3 + 9 + 3 + 1 + 1 + 1) * f4
    p2 += B * T * 3 * f4                   # output
    return dict(pass1_bytes=int(p1), pass2_bytes=int(p2), total_bytes=int(p1 + p2))
