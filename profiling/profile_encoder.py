"""Find out *why* `encode_taxels` is slow, rather than asserting a reason.

HYPOTHESES.md H2 is resolved: the naive splat encoder is 77.1% of step time at
N=4096. That tells us the encoder is the bottleneck; it does not tell us what
kind of bottleneck. The next kernel (Triton, then CUDA) has to be designed
against a *mechanism*, and the four candidate mechanisms want opposite designs:

  * bandwidth-bound      -> stop materialising (B, C, T, 3) intermediates;
                            recompute in registers (H6's gather reformulation)
  * compute-bound        -> the exp/einsum arithmetic is irreducible, so the win
                            is sparsity (only ~4.3 of 48 contact slots are live)
  * latency/occupancy    -> the shapes are fine, the launch config is not
  * kernel-launch bound  -> the problem is the *number* of torch ops, and any
                            single fused kernel wins regardless of its internals

So this script measures, in one place:
  1. a per-op CUDA table from `torch.profiler` (which op, how many launches,
     how much self CUDA time) at a given N;
  2. the analytic bytes-moved and FLOPs per call, derived op by op from the
     shapes in `src/encoders/taxel_torch.py`;
  3. an empirical DRAM bandwidth ceiling for this GB10, because published peak
     numbers for the Spark's LPDDR5X are not something to take on faith, and a
     roofline drawn against a wrong ridge point is worse than no roofline;
  4. a `--mode ncu` entry point that runs exactly one encoder call between
     `cudaProfilerStart/Stop`, so Nsight Compute profiles the encoder and not
     the warmup.

INPUTS ARE SYNTHETIC BY DEFAULT, and that is a deliberate, defensible choice:
the baseline encoder is dense and branchless. Every kernel it launches touches
all B*C*T elements no matter what the data says; `torch.where` is a select, not
a branch, and the masks change values, not work. Only `exp` has any data
dependence at all and that is at the ULP level. `--mode real` exists to confirm
that claim against the actual MuJoCo Warp contact stream rather than assume it.

SAFETY: this box has unified CPU/GPU memory (~121 GB shared). Always run under
    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
        .venv/bin/python profiling/profile_encoder.py --num-envs 1024
N=4096 allocates ~4.1 GB of torch tensors; under `ncu`, kernel replay multiplies
that, which is why the ncu path defaults to N=1024.

Outputs land in profiling/ as `optable_N{N}.txt` and `summary_N{N}.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless box; must precede mujoco import

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "profiling"
for _extra in (ROOT / "src", ROOT / "explore", ROOT / "benchmarks",
               ROOT / "third_party" / "leapXELA_model"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from encoders.taxel_torch import N_TAXELS, encode_taxels  # noqa: E402

# Must match benchmarks/harness.py or the measurement is of a different encoder.
NCONMAX_PER_ENV = 48        # C
KERNEL_SIGMA = 0.0035
KERNEL_CUTOFF = 0.01
LIVE_CONTACTS_PER_ENV = 4.3  # measured, benchmarks/results/scale_sweep_splat.csv


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #


def _rand_rotations(shape, gen, dev, dt) -> torch.Tensor:
    """Uniform random rotation matrices, built from quaternions.

    Deliberately NOT `torch.linalg.qr`: that pulls in `libtorch_cuda_linalg.so`,
    which fails to dlopen under `ncu`'s injection, and the profiling run must not
    depend on a library the profiler cannot load. Quaternion -> matrix is plain
    elementwise arithmetic, so it works everywhere and is exactly orthonormal.
    """
    q = torch.randn((*shape, 4), generator=gen, device=dev, dtype=dt)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return r.view(*shape, 3, 3)


def make_inputs(B: int, C: int = NCONMAX_PER_ENV, T: int = N_TAXELS, seed: int = 0):
    """Padded (B, C, ...) batch with the *statistics* of the real one.

    What is reproduced from the real workload (and why it could matter):
      * only ~4.3 of the 48 contact slots per env are valid -- the rest are
        padding. The dense encoder still multiplies through them, which is
        precisely the waste a gather kernel would skip, so the ratio has to be
        right for any projected speedup to mean anything.
      * contacts sit within a few mm of some taxel, so a realistic fraction of
        the Gaussian weights survive the cutoff (an all-zero weight field would
        make `exp` and the reductions unrepresentatively cheap on some hardware).
      * `site_mat` is a real rotation matrix, not noise, so the einsums see
        normal-magnitude values and no denormals.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = torch.device("cuda"), torch.float32

    # 12 taxel-carrying bodies, taxels grouped contiguously as in the layout.
    n_bodies = 12
    taxel_body = torch.arange(T, device=dev) * n_bodies // T

    site_pos = torch.rand((B, T, 3), generator=g, device=dev, dtype=dt) * 0.1
    site_mat = _rand_rotations((B, T), g, dev, dt)

    # Live contacts land on a random taxel of the body they touch, offset by
    # ~sigma, so the splat has real support. Dead slots are pure padding.
    n_live = int(round(LIVE_CONTACTS_PER_ENV * B))
    contact_valid = torch.zeros((B, C), dtype=torch.bool, device=dev)
    flat = torch.randperm(B * C, generator=g, device=dev)[:n_live]
    contact_valid.view(-1)[flat] = True

    anchor = torch.randint(0, T, (B, C), generator=g, device=dev)
    contact_pos = torch.gather(
        site_pos, 1, anchor.unsqueeze(-1).expand(B, C, 3)
    ) + torch.randn((B, C, 3), generator=g, device=dev, dtype=dt) * KERNEL_SIGMA
    contact_body = torch.where(
        contact_valid, taxel_body[anchor], torch.full_like(anchor, -1)
    )

    contact_frame = _rand_rotations((B, C), g, dev, dt)
    contact_force = torch.randn((B, C, 6), generator=g, device=dev, dtype=dt)
    contact_sign = torch.where(
        torch.rand((B, C), generator=g, device=dev) > 0.5, 1.0, -1.0
    ).to(dt)

    return dict(
        contact_pos=contact_pos.contiguous(),
        contact_frame=contact_frame.contiguous(),
        contact_force=contact_force.contiguous(),
        contact_sign=contact_sign,
        contact_body=contact_body,
        contact_valid=contact_valid,
        site_pos=site_pos.contiguous(),
        site_mat=site_mat.contiguous(),
        taxel_body=taxel_body,
        kernel_sigma=KERNEL_SIGMA,
        kernel_cutoff=KERNEL_CUTOFF,
    )


def call(inputs):
    return encode_taxels(**inputs)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def time_call(inputs, iters: int = 20, warmup: int = 5) -> dict:
    """Median wall time per call, CUDA-event timed with no profiler attached.

    This is the number every profiled number is sanity-checked against: if the
    profiler's total self-CUDA-time is far from this, the profiler is lying (or
    at least distorting), and that has to be said out loud rather than papered
    over.
    """
    for _ in range(warmup):
        call(inputs)
    torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        call(inputs)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return dict(
        ms_median=ts[len(ts) // 2], ms_min=ts[0], ms_max=ts[-1],
        ms_mean=float(np.mean(ts)), iters=iters,
    )


# --------------------------------------------------------------------------- #
# Cost model: bytes and FLOPs per op, joined against the profiler
# --------------------------------------------------------------------------- #


def cost_model(B: int, C: int = NCONMAX_PER_ENV, T: int = N_TAXELS) -> dict:
    """Expected DRAM traffic and FLOPs for every op `encode_taxels` launches.

    Keyed by `(aten name, input shapes)` so it can be JOINED against
    `torch.profiler`'s shape-grouped table. That join is the point: it turns
    "the encoder moves a lot of memory" into a per-kernel achieved-bandwidth
    number, which is the only way to tell a bandwidth-bound kernel (running at
    the DRAM ceiling) from a badly-shaped one (running at a third of it and
    nowhere near compute peak either). Any profiler row the model does not know
    about is reported as `UNMODELLED` rather than dropped, so the totals cannot
    quietly lose time.

    Counting rules, stated so they can be argued with:
      * bytes = compulsory DRAM traffic for an *unfused* execution: every
        intermediate is written out and read back, because that is what eager
        PyTorch does. Small operands (contact_*, a few MB) are counted once even
        though they are re-read; they are rounding error.
      * a stride-3 read of `local[..., i]` is charged the FULL (B,C,T,3) tensor.
        DRAM moves 32-byte sectors and a stride-12-byte access pattern touches
        every sector, so charging only the useful third would invent bandwidth
        that was never available. Marked `sector waste` below.
      * FLOPs: 1 per mul/add/sub/compare/divide. `exp` counted as 1, which
        *under*-counts it (it is a multi-instruction MUFU sequence) -- chosen
        deliberately so the compute side of the roofline is not flattered.
    """
    n = B * C * T
    f4, b1 = 4, 1
    m: dict[tuple, dict] = {}

    def add(name, shapes, label, read, write, flops, note=""):
        m[(name, str(shapes))] = dict(
            label=label, read_bytes=read, write_bytes=write,
            bytes=read + write, flops=flops, note=note,
        )

    # -- force_world = einsum("bcji,bcj->bci") * sign ------------------------- #
    # Lowered to a batched 3x3-by-3 gemv. Negligible, listed for completeness.
    add("aten::bmm", [[B * C, 3, 3], [B * C, 3, 1]], "einsum: force_world",
        B * C * 12 * f4, B * C * 3 * f4, B * C * 15)
    add("aten::mul", [[B, C, 3], [B, C, 1]], "force_world *= sign",
        B * C * 4 * f4, B * C * 3 * f4, B * C * 3)

    # -- rel = contact_pos[:, :, None] - site_pos[:, None] -> (B, C, T, 3) ----- #
    add("aten::sub", [[B, C, 1, 3], [B, 1, T, 3], []], "rel = broadcast sub",
        (B * C * 3 + B * T * 3) * f4, n * 3 * f4, n * 3,
        "materialises 3n floats from 3(BC+BT): zero reuse")

    # -- local = einsum("btji,bctj->bcti") ------------------------------------ #
    # torch lowers this to permute(rel) -> contiguous copy -> batched gemm with
    # B*T batches of (C x 3) @ (3 x 3). The copy and the gemm are separate rows.
    add("aten::copy_", [[B, T, C, 3, 1], [B, T, C, 3, 1], []],
        "einsum(local): transpose rel to (B,T,C,3)",
        n * 3 * f4, n * 3 * f4, 0, "pure data movement, no arithmetic")
    add("aten::bmm", [[B * T, C, 3], [B * T, 3, 3]], "einsum(local): batched 3x3 gemm",
        (n * 3 + B * T * 9) * f4, n * 3 * f4, n * 15,
        "B*T batches of a 48x3x3 gemm -- K=3")

    # -- dist_sq, weights ----------------------------------------------------- #
    add("aten::pow", [[B, C, T], []], "local[...,i]**2 (x2)",
        n * 3 * f4, n * f4, n, "stride-3 read: sector waste, charged full tensor")
    add("aten::add", [[B, C, T], [B, C, T], []], "dist_sq = x^2 + y^2",
        n * 2 * f4, n * f4, n)
    add("aten::neg", [[B, C, T]], "-dist_sq", n * f4, n * f4, n)
    add("aten::div", [[B, C, T], []], "/(2 sigma^2)", n * f4, n * f4, n)
    add("aten::exp", [[B, C, T]], "exp", n * f4, n * f4, n)

    # -- the two cutoff gates -------------------------------------------------- #
    add("aten::gt", [[B, C, T], []], "cutoff compares (x2)", n * f4, n * b1, n)
    add("aten::fill_", [[B, C, T], []], "zeros_like for where (x2)", 0, n * f4, 0,
        "an entire 4n-byte buffer of zeros, written then read, twice")
    add("aten::where", [[B, C, T], [B, C, T], [B, C, T]], "where(gate, 0, w) (x2)",
        (n * b1 + n * f4 + n * f4), n * f4, 0)
    add("aten::abs", [[B, C, T], [0]], "abs(local[...,2])",
        n * 3 * f4, n * f4, n, "stride-3 read: sector waste")

    # -- body / validity mask -------------------------------------------------- #
    add("aten::eq", [[B, C, 1], [1, 1, T]], "body_mask", (B * C + T) * 8, n * b1, n)
    add("aten::bitwise_and", [[B, C, T], [B, C, 1]], "keep = mask & valid",
        (n + B * C) * b1, n * b1, n)
    add("aten::copy_", [[B, C, T], [B, C, T], []], "keep.to(float32)",
        n * b1, n * f4, 0)
    add("aten::mul", [[B, C, T], [B, C, T]], "weights *= keep", n * 2 * f4, n * f4, n)

    # -- per-contact normalisation --------------------------------------------- #
    add("aten::sum", [[B, C, T], [], [], []], "weight_sum over taxels",
        n * f4, B * C * f4, n)
    add("aten::div", [[B, C, T], [B, C, 1]], "weights /= denom",
        (n + B * C) * f4, n * f4, n)

    # -- force_local = einsum("btji,bcj->bcti") --------------------------------- #
    # Lowered to one gemm per env: (C x 3) @ (3 x 3T). Reads almost nothing and
    # writes 3n floats -- the worst input:output ratio in the function.
    add("aten::copy_", [[B, 3, T, 3, 1], [B, 3, T, 3, 1], []],
        "einsum(force_local): transpose site_mat",
        B * T * 9 * f4, B * T * 9 * f4, 0)
    add("aten::bmm", [[B, C, 3], [B, 3, T * 3]], "einsum(force_local): gemm",
        (B * C * 3 + B * T * 9) * f4, n * 3 * f4, n * 15,
        "expands (B,C,3) + (B,T,9) into 3n floats")

    # -- accumulate ------------------------------------------------------------- #
    add("aten::mul", [[B, C, T, 1], [B, C, T, 3]], "weights * force_local",
        (n + n * 3) * f4, n * 3 * f4, n * 3)
    add("aten::sum", [[B, C, T, 3], [], [], []], "sum over contacts",
        n * 3 * f4, B * T * 3 * f4, n * 3, "reduction over a non-innermost axis")
    add("aten::mul", [[B, T, 3], [3]], "out *= [1,1,-1]",
        B * T * 3 * f4, B * T * 3 * f4, B * T * 3)

    return m


def design_floor(B: int, C: int = NCONMAX_PER_ENV, T: int = N_TAXELS,
                 live_per_env: float = LIVE_CONTACTS_PER_ENV,
                 bodies: int = 12) -> dict:
    """The traffic and FLOP floors any reformulation is bounded by.

    A fused kernel that keeps every intermediate in registers has to move only
    the *inputs and the output* -- nothing else. That is the roofline floor, and
    it is what makes the difference between "shave 30%" and "two orders of
    magnitude" concrete.

    Three FLOP floors, because they correspond to three different designs:
      dense    -- loop all C=48 slots per env (simplest gather kernel)
      live     -- loop only the live contacts (needs a per-world contact count)
      bodysparse -- live contacts x only the taxels of the body touched, which
                    is what the `body_mask` in the encoder already implies.
                    Needs taxels grouped by body, which they already are.
    """
    f4 = 4
    inputs = (B * T * 3          # site_pos
              + B * T * 9        # site_mat
              + B * C * 3        # contact_pos
              + B * C * 9        # contact_frame
              + B * C * 3        # contact_force[:3]
              + B * C * 2) * f4  # sign + body
    output = B * T * 3 * f4
    per_pair = 53                # FLOPs per (env, contact, taxel) pair, as modelled
    return dict(
        min_bytes=inputs + output,
        input_bytes=inputs, output_bytes=output,
        flops_dense=B * C * T * per_pair,
        flops_live=int(B * live_per_env * T * per_pair),
        flops_bodysparse=int(B * live_per_env * (T / bodies) * per_pair),
        pairs_dense=B * C * T,
        pairs_live=int(B * live_per_env * T),
        pairs_bodysparse=int(B * live_per_env * (T / bodies)),
    )


def cost_totals(model: dict, counts: dict | None = None) -> dict:
    """Sum the model. `counts` supplies how many times each entry actually ran
    (the two `pow`s and two `where`s share one model entry)."""
    r = w = f = 0
    for k, v in model.items():
        c = (counts or {}).get(k, 1)
        r += v["read_bytes"] * c
        w += v["write_bytes"] * c
        f += v["flops"] * c
    return dict(read_bytes=r, write_bytes=w, total_bytes=r + w, flops=f,
                arithmetic_intensity=f / (r + w))

# --------------------------------------------------------------------------- #
# Empirical bandwidth ceiling
# --------------------------------------------------------------------------- #


def measure_bandwidth(size_mb: int = 1024, iters: int = 30) -> dict:
    """Empirical DRAM ceiling: the ridge point of the roofline, measured here.

    NVIDIA does not publish an authoritative STREAM number for the GB10's
    unified LPDDR5X that I am willing to draw a roofline against, and on a
    unified-memory part the "DRAM" the encoder hits is the same memory the host
    uses. So measure it: three access patterns, take the best as the ceiling.

      copy   : read 1, write 1  -- the classic saturating pattern
      read   : read 1, write ~0 -- a full reduction, no write traffic
      triad  : read 2, write 1  -- what a real elementwise op looks like

    Buffers are sized well past the 24 MB L2 so nothing is cached.
    """
    n = size_mb * 1024 * 1024 // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32).normal_()
    b = torch.empty_like(a)
    c = torch.empty_like(a)

    def bench(fn, bytes_moved):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return bytes_moved * iters / (s.elapsed_time(e) / 1e3) / 1e9

    nb = n * 4
    res = dict(
        copy_GBps=bench(lambda: b.copy_(a), 2 * nb),
        read_GBps=bench(lambda: torch.sum(a), nb),
        triad_GBps=bench(lambda: torch.add(a, b, out=c), 3 * nb),
        buffer_MB=size_mb,
    )
    del a, b, c
    torch.cuda.empty_cache()
    res["ceiling_GBps"] = max(res["copy_GBps"], res["read_GBps"], res["triad_GBps"])
    return res


def measure_compute_ceiling(n: int = 8192, iters: int = 20) -> dict:
    """Empirical FP32 ceiling: the other axis of the roofline, also measured.

    Nsight Compute cannot read hardware counters on this box (the driver is
    built with `NVreg_RestrictProfilingToAdminUsers=1` and there is no root), so
    "% of peak" has to come from a measured ceiling rather than from a counter or
    a datasheet. A large square SGEMM with TF32 explicitly OFF is the standard
    stand-in: it is the densest FP32 FMA stream this device will actually run.

    Reported as the *achievable* ceiling, not the theoretical one. Theoretical
    for reference: 48 SMs x 128 FP32 lanes x 2 flop x 2.418 GHz = 29.7 TFLOP/s.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 would not be FP32 peak
    try:
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            a @ b
        e.record()
        torch.cuda.synchronize()
        tflops = 2 * n ** 3 * iters / (s.elapsed_time(e) / 1e3) / 1e12
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        del a, b
        torch.cuda.empty_cache()
    p = torch.cuda.get_device_properties(0)
    theo = p.multi_processor_count * 128 * 2 * (p.clock_rate / 1e6) / 1e3
    return dict(sgemm_TFLOPs=tflops, theoretical_fp32_TFLOPs=theo, gemm_n=n)


def measure_launch_overhead(iters: int = 2000) -> dict:
    """Cost of issuing a trivial kernel, to price the launch-bound hypothesis.

    A null kernel (elementwise op on 1 element) measures launch + dispatch, and
    that includes the Python/ATen dispatch above it -- which is the honest thing
    to price, because the encoder pays that too. Reported both as end-to-end
    (queue is drained) and as sustained issue rate (queue stays full), because
    the encoder's launches pipeline and only the second number bounds it.
    """
    x = torch.ones(1, device="cuda")
    for _ in range(100):
        x.add_(1.0)
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        x.add_(1.0)
    e.record()
    torch.cuda.synchronize()
    pipelined_us = s.elapsed_time(e) * 1e3 / iters

    import time as _t
    t0 = _t.perf_counter()
    for _ in range(200):
        x.add_(1.0)
        torch.cuda.synchronize()
    serial_us = (_t.perf_counter() - t0) * 1e6 / 200
    return dict(pipelined_launch_us=pipelined_us, serialised_launch_us=serial_us)


# --------------------------------------------------------------------------- #
# torch.profiler
# --------------------------------------------------------------------------- #


def profile_ops(inputs, B: int, n_calls: int = 5,
                trace: pathlib.Path | None = None) -> dict:
    """Per-op CUDA table, joined against the cost model.

    ATen rows and CUDA-kernel rows BOTH carry self-device time in
    `key_averages()`, and they are the same time counted twice (one names the
    op, the other the kernel it launched). Summing everything therefore doubles
    the total -- the ATen rows alone reconcile exactly with the profiler's own
    "Self CUDA time total", so those are what the tally uses.

    Kernel launches are counted separately from `cudaLaunchKernel` /
    `cuLaunchKernel`, because ops and kernels are not 1:1 (one `aten::bmm` here
    fans out into six cutlass launches) and the launch-overhead hypothesis has
    to be priced against the real launch count.
    """
    from collections import defaultdict

    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        call(inputs)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_calls):
            call(inputs)
        torch.cuda.synchronize()

    ka = prof.key_averages(group_by_input_shape=True)
    table_op = prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=60, max_name_column_width=55
    )
    table_shape = ka.table(
        sort_by="self_device_time_total", row_limit=50, max_name_column_width=45
    )

    model = cost_model(B)
    launches = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cuLaunchKernel"))

    rows, counts = [], defaultdict(int)
    total_us = 0.0
    for e in ka:
        if not e.key.startswith("aten::") or e.self_device_time_total <= 0:
            continue
        total_us += e.self_device_time_total
        key = (e.key, str(list(e.input_shapes)))
        mv = model.get(key)
        counts[key] += e.count // n_calls
        us = e.self_device_time_total / n_calls
        rows.append(dict(
            op=e.key,
            shapes=str(list(e.input_shapes)),
            label=mv["label"] if mv else "UNMODELLED",
            calls_per_encode=e.count / n_calls,
            us_per_call=us,
            bytes=(mv["bytes"] * (e.count // n_calls)) if mv else None,
            flops=(mv["flops"] * (e.count // n_calls)) if mv else None,
            achieved_GBps=(mv["bytes"] * (e.count // n_calls) / (us / 1e6) / 1e9)
            if mv else None,
            note=mv["note"] if mv else "",
        ))
    rows.sort(key=lambda r: -r["us_per_call"])

    unmodelled_us = sum(r["us_per_call"] for r in rows if r["label"] == "UNMODELLED")
    totals = cost_totals(model, counts)

    if trace is not None:
        prof.export_chrome_trace(str(trace))

    return dict(
        table_op=table_op, table_shape=table_shape, n_calls=n_calls,
        device_us_per_call=total_us / n_calls,
        aten_ops_per_call=sum(r["calls_per_encode"] for r in rows),
        kernel_launches_per_call=launches / n_calls,
        rows=rows,
        unmodelled_us_per_call=unmodelled_us,
        modelled_totals=totals,
    )


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #


def mode_bench(args) -> int:
    B = args.num_envs
    inputs = make_inputs(B, seed=args.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timing = time_call(inputs, iters=args.iters)
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    prof = profile_ops(
        inputs, B, n_calls=args.profile_calls,
        trace=(OUT / "nsight" / f"trace_N{B}.json") if args.trace else None,
    )
    tot = prof["modelled_totals"]

    bw = measure_bandwidth() if args.bandwidth else None
    launch = measure_launch_overhead() if args.bandwidth else None

    ms = timing["ms_median"]
    eff_bw = tot["total_bytes"] / (ms / 1e3) / 1e9
    eff_fl = tot["flops"] / (ms / 1e3) / 1e12

    hdr = (f"encode_taxels op breakdown, N={B}, C={NCONMAX_PER_ENV}, T={N_TAXELS}\n"
           f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}\n\n"
           f"unprofiled CUDA-event time per call: {ms:.3f} ms "
           f"(min {timing['ms_min']:.3f}, max {timing['ms_max']:.3f})\n"
           f"profiler self-device time per call:  "
           f"{prof['device_us_per_call'] / 1e3:.3f} ms  "
           f"(ATen rows only; kernel rows are the same time under another name)\n"
           f"CUDA kernel launches per call:       "
           f"{prof['kernel_launches_per_call']:.0f}\n"
           f"torch peak allocated: {peak_gb:.3f} GB\n"
           f"time in UNMODELLED ops: {prof['unmodelled_us_per_call'] / 1e3:.3f} ms\n\n")

    # The joined table: measured time next to modelled traffic, so every row
    # carries its own achieved bandwidth. This is the evidence for the verdict.
    join = ["=== measured time joined with the cost model ===",
            f"{'op':<18}{'what':<40}{'x':>4}{'ms':>9}{'%':>7}{'MB':>10}{'GB/s':>8}  note",
            "-" * 130]
    for r in prof["rows"]:
        join.append(
            f"{r['op'][:17]:<18}{r['label'][:39]:<40}{r['calls_per_encode']:>4.0f}"
            f"{r['us_per_call'] / 1e3:>9.3f}"
            f"{100 * r['us_per_call'] / prof['device_us_per_call']:>7.1f}"
            f"{(r['bytes'] or 0) / 1e6:>10.1f}"
            f"{(r['achieved_GBps'] or 0):>8.1f}  {r['note']}")
    join.append("-" * 130)
    join.append(f"{'TOTAL':<18}{'':<40}{'':>4}{ms:>9.3f}{100.0:>7.1f}"
                f"{tot['total_bytes'] / 1e6:>10.1f}{eff_bw:>8.1f}")
    join.append(f"\nmodelled FLOPs/call: {tot['flops'] / 1e9:.3f} GFLOP  "
                f"-> {eff_fl * 1e3:.1f} GFLOP/s achieved")
    join.append(f"arithmetic intensity: {tot['arithmetic_intensity']:.3f} FLOP/byte")
    join_txt = "\n".join(join)

    txt = OUT / f"optable_N{B}.txt"
    txt.write_text(hdr + join_txt
                   + "\n\n\n=== grouped by ATen op (raw profiler) ===\n"
                   + prof["table_op"]
                   + "\n\n=== grouped by op + input shape (raw profiler) ===\n"
                   + prof["table_shape"] + "\n")
    print(f"[profile] wrote {txt}")

    summary = dict(
        num_envs=B, C=NCONMAX_PER_ENV, T=N_TAXELS,
        device=torch.cuda.get_device_name(0),
        timing=timing,
        torch_peak_gb=peak_gb,
        profiler_device_ms_per_call=prof["device_us_per_call"] / 1e3,
        aten_ops_per_call=prof["aten_ops_per_call"],
        kernel_launches_per_call=prof["kernel_launches_per_call"],
        unmodelled_ms_per_call=prof["unmodelled_us_per_call"] / 1e3,
        rows=prof["rows"],
        modelled_totals=tot,
        effective_GBps=eff_bw,
        effective_TFLOPs=eff_fl,
        bandwidth=bw,
        launch=launch,
    )
    js = OUT / f"summary_N{B}.json"
    js.write_text(json.dumps(summary, indent=2))
    print(f"[profile] wrote {js}\n")
    print(join_txt)
    if bw:
        print(f"\nceiling: copy {bw['copy_GBps']:.1f}  read {bw['read_GBps']:.1f}  "
              f"triad {bw['triad_GBps']:.1f} GB/s  "
              f"-> {100 * eff_bw / bw['ceiling_GBps']:.0f}% of ceiling")
        print(f"launch : {launch['pipelined_launch_us']:.2f} us pipelined x "
              f"{prof['kernel_launches_per_call']:.0f} launches = "
              f"{launch['pipelined_launch_us'] * prof['kernel_launches_per_call'] / 1e3:.3f} ms "
              f"({100 * launch['pipelined_launch_us'] * prof['kernel_launches_per_call'] / 1e3 / ms:.1f}% of the call)")
    return 0


def mode_fusion(args) -> int:
    """Controlled experiment: hold the arithmetic fixed, remove the intermediates.

    This is the test that distinguishes the four candidate bottlenecks from each
    other without needing hardware counters. `torch.compile` (inductor) fuses the
    elementwise chain into a handful of Triton kernels; it does NOT change a
    single FLOP, does NOT change occupancy strategy, and does NOT reduce the
    number of (env, contact, taxel) pairs visited. The only thing it removes is
    round trips to DRAM.

      * if the encoder is bandwidth-bound, fusing wins large and the win tracks
        the traffic removed;
      * if it is compute-bound, fusing changes nothing;
      * if it is launch-bound, fusing wins a fixed ~4 us per launch removed,
        which at 37 launches is 0.16 ms and invisible.

    It is also a lower bound on the hand-written kernel: inductor still has to
    materialise anything it cannot fuse (the two einsums), so a real gather
    kernel should beat whatever this shows.
    """
    B = args.num_envs
    inputs = make_inputs(B, seed=args.seed)

    ref = call(inputs)
    eager = time_call(inputs, iters=args.iters)

    compiled = torch.compile(encode_taxels, dynamic=False)

    def call_c(inp):
        return compiled(**inp)

    got = call_c(inputs)
    err = float((got - ref).abs().max().item())

    for _ in range(5):
        call_c(inputs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        call_c(inputs)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    fused_ms = ts[len(ts) // 2]

    # How many kernels did inductor end up launching, and how much traffic does
    # the speedup imply it removed?
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call_c(inputs)
        torch.cuda.synchronize()
    launches = sum(e.count for e in prof.key_averages()
                   if e.key in ("cudaLaunchKernel", "cuLaunchKernel"))

    tot = cost_totals(cost_model(B), _model_counts(B))
    floor = design_floor(B)
    out = dict(
        num_envs=B, eager_ms=eager["ms_median"], fused_ms=fused_ms,
        speedup=eager["ms_median"] / fused_ms,
        max_abs_err_vs_eager=err,
        fused_kernel_launches=launches,
        eager_modelled_bytes=tot["total_bytes"],
        implied_fused_GBps=tot["total_bytes"] / (fused_ms / 1e3) / 1e9,
        floor_bytes=floor["min_bytes"],
        floor_ms_at_240GBps=floor["min_bytes"] / 240e9 * 1e3,
    )
    (OUT / f"fusion_N{B}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def _model_counts(B: int) -> dict:
    """Multiplicities for the model entries that run more than once per call."""
    n = cost_model(B)
    counts = {k: 1 for k in n}
    for name, shapes in (("aten::pow", [[B, NCONMAX_PER_ENV, N_TAXELS], []]),
                         ("aten::gt", [[B, NCONMAX_PER_ENV, N_TAXELS], []]),
                         ("aten::fill_", [[B, NCONMAX_PER_ENV, N_TAXELS], []]),
                         ("aten::where", [[B, NCONMAX_PER_ENV, N_TAXELS]] * 3)):
        counts[(name, str(shapes))] = 2
    return counts


def mode_ncu(args) -> int:
    """One encoder call inside a cudaProfilerStart/Stop window, nothing else.

    Run under:
      ncu --profile-from-start off --set full -o profiling/nsight/enc_N1024 \\
          <python> profiling/profile_encoder.py --mode ncu --num-envs 1024
    """
    B = args.num_envs
    inputs = make_inputs(B, seed=args.seed)
    for _ in range(args.warmup):
        call(inputs)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    call(inputs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"[ncu] profiled one encode_taxels call at N={B}")
    return 0


def mode_bandwidth(args) -> int:
    """Measure both roofline axes and the launch cost, and record the ridge point."""
    bw = measure_bandwidth(size_mb=args.bw_mb)
    comp = measure_compute_ceiling()
    launch = measure_launch_overhead()
    out = dict(bandwidth=bw, compute=comp, launch=launch,
               ridge_point_flop_per_byte=comp["sgemm_TFLOPs"] * 1e12
               / (bw["ceiling_GBps"] * 1e9),
               device=torch.cuda.get_device_name(0),
               sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
    (OUT / "bandwidth.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def mode_real(args) -> int:
    """Cross-check: drive the encoder from the real MuJoCo Warp contact stream.

    Confirms (or refutes) the claim that synthetic inputs are performance-
    equivalent, and measures how much of the harness's `encoder_ms_per_step`
    is `encode_taxels` itself versus the contact bucketing around it.
    """
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING
    import mujoco as mj
    import mujoco_warp as mjw

    import harness as H

    B = args.num_envs
    mjm = H.build_scene_model(H.DEFAULT_SCENE, "splat")
    mjd = mj.MjData(mjm)
    mj.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=B,
                     naconmax=H.NCONMAX_PER_ENV * B, njmax=H.NJMAX_PER_ENV)
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()), wp.get_device())
    enc = H._attach_mjw(H.SplatEncoder(mjm, m, d, B), mjw)

    lo, hi = mjm.actuator_ctrlrange[:, 0], mjm.actuator_ctrlrange[:, 1]
    d.ctrl.assign(np.tile((lo + H.CLOSE_FRACTION * (hi - lo)).astype(np.float32), (B, 1)))
    for _ in range(H.SETTLE_STEPS):
        mjw.step(m, d)
    wp.synchronize()

    # Freeze one real batch and time the encoder on it, so the comparison with
    # the synthetic path is like for like (same call, different data).
    pos, frame, force, sign, body, valid = enc._batch()
    real = dict(
        contact_pos=pos, contact_frame=frame, contact_force=force,
        contact_sign=sign, contact_body=body, contact_valid=valid,
        site_pos=enc.v_site_xpos[:, enc.site_ids].contiguous(),
        site_mat=enc.v_site_xmat[:, enc.site_ids].contiguous(),
        taxel_body=enc.taxel_body,
        kernel_sigma=H.KERNEL_SIGMA, kernel_cutoff=H.KERNEL_CUTOFF,
    )
    t_real = time_call(real, iters=args.iters)
    t_syn = time_call(make_inputs(B, seed=args.seed), iters=args.iters)

    # Full encode() = contact_force + _batch + encode_taxels; the difference is
    # the bucketing overhead the harness attributes to "encoder".
    for _ in range(5):
        enc.encode()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(args.iters):
        enc.encode()
    e.record()
    torch.cuda.synchronize()
    full_ms = s.elapsed_time(e) / args.iters

    live = int(valid.sum().item())
    out = dict(num_envs=B, real=t_real, synthetic=t_syn,
               full_encode_ms=full_ms,
               batching_ms=full_ms - t_real["ms_median"],
               live_contacts=live, live_per_env=live / B,
               ratio_real_over_synth=t_real["ms_median"] / t_syn["ms_median"])
    (OUT / f"real_check_N{B}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--mode",
                    choices=["bench", "ncu", "fusion", "bandwidth", "real"],
                    default="bench")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile-calls", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bw-mb", type=int, default=1024)
    ap.add_argument("--bandwidth", action="store_true",
                    help="also measure the empirical DRAM ceiling and launch cost")
    ap.add_argument("--trace", action="store_true", help="export a chrome trace")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    (OUT / "nsight").mkdir(exist_ok=True)
    return dict(bench=mode_bench, ncu=mode_ncu, fusion=mode_fusion,
                bandwidth=mode_bandwidth, real=mode_real)[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
