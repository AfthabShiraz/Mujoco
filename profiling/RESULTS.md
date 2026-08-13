# Why `encode_taxels` is slow

**Measured 2026-08-13 on the DGX Spark (NVIDIA GB10, 48 SMs, sm_121, unified
LPDDR5X). torch 2.13.0+cu130, CUDA 13.0.**
Reproduce with `profiling/profile_encoder.py` (every run memory-capped —
see the safety note at the bottom).

## Answer, up front

**Bandwidth-bound, decisively — and worse than that, most of the bandwidth is
spent on data that is provably discarded.**

- Arithmetic intensity is **0.192 FLOP/byte**. The measured ridge point of this
  GPU is **70.7 FLOP/byte**. The encoder sits **368× to the left of the ridge**.
- It achieves **35.3 GFLOP/s**, which is **0.19%** of the measured 18.44 TFLOP/s
  SGEMM ceiling. Compute is not the limit and is not close to being the limit.
- Kernel-launch overhead is **0.18%** of the call. Not the limit either.
- The direct experiment: `torch.compile` fuses the elementwise chain, changes
  **zero FLOPs**, and makes it **2.40× faster** with bit-comparable output
  (max abs err 4.8e-7). The only variable it changed was DRAM round trips.

The one part that is *not* bandwidth-bound is the `einsum` → `bmm` lowering
(34.8% of the call), and that is a separate pathology: it runs at 28% of DRAM
peak *and* 0.14% of compute peak simultaneously. It is bound by neither, because
cuBLAS is being asked to run 1.5M batched 48×3×3 SGEMMs with K=3.

---

## 0. What was measured, and what could not be

| Instrument | Status |
|---|---|
| `torch.profiler` per-op CUDA table | ✅ used, N=1024 and N=4096 |
| `nsys` timeline + kernel summary | ✅ `profiling/nsight/enc_N{1024,4096}.nsys-rep` |
| `ncu` hardware counters | ❌ **blocked** — `ERR_NVGPUCTRPERM` |
| Empirical DRAM ceiling (saturating copy) | ✅ substituted for the ncu counters |
| Empirical FP32 ceiling (SGEMM, TF32 off) | ✅ substituted for the ncu counters |

**`ncu` could not read a single counter on this box.** Every section, including
`LaunchStats` and `Occupancy`, returns `ERR_NVGPUCTRPERM`: the driver is built
with `NVreg_RestrictProfilingToAdminUsers=1` and there is no root on this
machine. So there is **no Nsight-reported limiter in this report, and no
Nsight-measured DRAM throughput** — those two deliverables could not be produced
as specified. Everything labelled "achieved GB/s" below is
`modelled bytes ÷ measured kernel time`, benchmarked against a bandwidth ceiling
I measured myself. That substitution is stated wherever the number appears; it
is weaker than a counter read, and the fusion experiment (§4) exists precisely
because it tests the same conclusion without needing counters.

Two `.nsys-rep` files are under `profiling/nsight/` (gitignored, intended), with
kernel summaries dumped alongside as `*_kernsum.txt`.

### Confidence in the numbers

- **Profiler overhead is negligible here.** Unprofiled CUDA-event median at
  N=4096: **108.753 ms**. Profiler self-device total: **108.911 ms**. 0.15%
  apart, so the op table is not distorted.
- **Synthetic inputs are performance-equivalent to real ones.** `--mode real`
  drives the encoder from the actual MuJoCo Warp contact stream at N=1024:
  **27.386 ms real vs 27.371 ms synthetic — a ratio of 1.0006.** As expected for
  a dense branchless formulation, but now checked rather than assumed.
- **`encode_taxels` really is the harness's encoder cost.** At N=4096 the CSV
  reports `encoder_ms_per_step = 112.26`; `encode_taxels` alone is **108.75 ms**
  (97%). At N=1024 the full `encode()` is 28.04 ms of which contact bucketing +
  `contact_force` is **0.65 ms (2.3%)**. The bucketing is not the problem; the
  harness docstring's claim that "the sort is not where the time goes" is
  confirmed.
- **Unmodelled ops account for 0.015 ms of 108.75.** Nothing was silently lost.
- **Run-to-run spread is about 0.5%.** Two independent N=4096 runs gave 108.75
  and 109.32 ms/call. Numbers below are quoted from the first run; treat the
  last significant figure as noise, and do not read anything into a difference
  smaller than ~1%.

---

## 1. Op-level breakdown

Full tables: `profiling/optable_N1024.txt`, `profiling/optable_N4096.txt`.
JSON: `profiling/summary_N{1024,4096}.json`.

### N=4096 — 108.75 ms/call, 58 kernel launches, 4.045 GB torch peak

| op | what | × | ms | % | MB moved | GB/s | % of ceiling |
|---|---|---|---|---|---|---|---|
| `bmm` | einsum(`local`): batched 3×3 gemm | 1 | 24.53 | 22.5 | 1790.7 | **73.0** | 28% |
| `pow` | `local[...,i]**2` | 2 | 9.61 | 8.8 | 2315.3 | 240.9 | 92% |
| `mul` | `weights * force_local` | 1 | 9.16 | 8.4 | 2025.8 | 221.3 | 85% |
| `where` | `where(gate, 0, w)` | 2 | 8.04 | 7.4 | 1881.1 | 233.9 | 90% |
| `copy_` | einsum(`local`): transpose `rel` | 1 | 7.63 | 7.0 | 1736.4 | 227.5 | 87% |
| `sum` | sum over contacts (`dim=1`) | 1 | 7.12 | 6.5 | 886.3 | **124.4** | 48% |
| `abs` | `abs(local[...,2])` | 1 | 4.93 | 4.5 | 1157.6 | 234.7 | 90% |
| `bmm` | einsum(`force_local`): gemm | 1 | 4.67 | 4.3 | 924.8 | 198.1 | 76% |
| `sub` | `rel` = broadcast sub | 1 | 4.53 | 4.2 | 888.7 | 196.3 | 75% |
| `mul` | `weights *= keep` | 1 | 3.90 | 3.6 | 868.2 | 222.9 | 85% |
| `add` | `dist_sq = x² + y²` | 1 | 3.65 | 3.3 | 868.2 | 238.0 | 91% |
| `fill_` | `zeros_like` for `where` | 2 | 2.98 | 2.7 | 578.8 | 194.5 | 75% |
| `gt` | cutoff compares | 2 | 2.95 | 2.7 | 723.5 | 245.6 | 94% |
| `div`/`exp`/`neg`/`div` | the Gaussian itself | 4 | 9.82 | 9.0 | 2316 | ~236 | 90% |
| everything else | masks, casts, `weight_sum`, small einsums | 12 | 5.41 | 5.0 | 1033 | ~191 | 73% |
| **TOTAL** | | **58** | **108.75** | 100 | **19994** | **183.9** | **71%** |

N=1024 is the same shape scaled by ~4 (27.28 ms, 5.00 GB, 37 launches, 183.3
GB/s) — the mechanism does not change with N, which matters because it means
the fix generalises.

**Grouped by mechanism (N=4096):**

| group | ms | % of call | aggregate GB/s |
|---|---|---|---|
| elementwise / broadcast / mask / reduce | 63.98 | 58.8% | **227.3 (87% of ceiling)** |
| `einsum` machinery (3 `bmm` + 2 transposes) | 37.81 | 34.8% | 120.9 (46%) |
| `sum` over the contact axis | 7.12 | 6.5% | 124.4 (48%) |

**Memory movement vs arithmetic.** Of the 108.75 ms, the ops that perform *no
useful arithmetic at all* — the two einsum transposes (8.10 ms), the two
`zeros_like` fills (2.98 ms), and the bool→float cast (1.66 ms) — are
**12.75 ms, 11.7%**. But that undersells it: the ops that *do* arithmetic are
still limited by their operand traffic, not their arithmetic. `exp` over 72M
elements takes **2.50 ms**, which is exactly the time to read and write 578 MB
at 232 GB/s. The transcendental is free; the load and the store are the cost.

**Launch count.** 58 CUDA kernel launches per call at N=4096 (37 at N=1024) from
71 ATen ops. Measured launch cost is 3.34 µs pipelined, so 58 launches ≈
**0.194 ms = 0.18% of the call**. Launch overhead is not a factor and any design
justified on "fewer kernel launches" is justified on the wrong grounds.

---

## 2. Roofline

### Bytes and FLOPs per call, derived from the code

`cost_model()` in `profile_encoder.py` derives, op by op, the compulsory DRAM
traffic of an unfused execution and joins it against the profiler by
`(op name, input shapes)`. Counting rules are in that function's docstring; two
are worth restating because they are the ones that could be argued with:

- A stride-3 read of `local[..., i]` is charged the **full** `(B,C,T,3)` tensor.
  DRAM moves 32-byte sectors and a 12-byte stride touches every sector, so
  charging only the useful third would invent bandwidth that never existed.
  (The measurement backs this: `pow` lands at 240.9 GB/s under this rule, i.e.
  at the ceiling. Under the naive rule it would read as 80 GB/s, implying a
  mystery inefficiency that does not exist.)
- `exp` is counted as **1 FLOP**, which under-counts it. Deliberate: inflating
  it would flatter the compute axis, and the whole point is not to do that.

**At N=4096, per call: 19.99 GB moved, 3.843 GFLOP.**
**Arithmetic intensity = 0.192 FLOP/byte.**

### The ceilings, measured on this machine

Published GB10 figures were not used. `--mode bandwidth` measures them:

| | measured | note |
|---|---|---|
| DRAM read (1 GB reduction) | **260.8 GB/s** | the ceiling used below |
| DRAM copy (read 1 + write 1) | 240.7 GB/s | |
| DRAM triad (read 2 + write 1) | 237.6 GB/s | closest to a real elementwise op |
| FP32 SGEMM, TF32 disabled, 8192³ | **18.44 TFLOP/s** | achievable FP32 ceiling |
| FP32 theoretical (48 SM × 128 × 2 × 2.418 GHz) | 29.71 TFLOP/s | for reference only |
| **Ridge point** | **70.7 FLOP/byte** | 18.44e12 ÷ 260.8e9 |

Unified memory caveat: on this part "DRAM" is the same LPDDR5X the host uses, so
the ceiling is a *shared* resource and 260.8 GB/s is what was available with the
box otherwise idle. Under a real training loop with the host busy it will be
lower, which makes the encoder's position worse, not better.

### Where the encoder sits

```
        AI = 0.192                    ridge = 70.7 FLOP/byte
             |                                    |
  <----------+------------------------------------+---------->
        368x to the left                     compute-bound
```

| | achieved | ceiling | fraction |
|---|---|---|---|
| DRAM throughput | **183.9 GB/s** | 260.8 GB/s | **71%** |
| FP32 throughput | **35.3 GFLOP/s** | 18,437 GFLOP/s | **0.19%** |

The encoder runs at 71% of the machine's memory bandwidth and 0.19% of its
arithmetic. The 29% bandwidth shortfall is not spread evenly — the elementwise
majority runs at **87%** of ceiling, and the whole shortfall is concentrated in
the two `bmm`s and the contact-axis reduction.

---

## 3. The top 3 costs, ranked, with the evidence

### 1. Materialising `(B, C, T, ·)` intermediates — 20.0 GB of DRAM traffic per step

**63.98 ms (58.8%) is spent in elementwise/mask/reduce kernels running at
227.3 GB/s aggregate, i.e. 87% of the measured 260.8 GB/s ceiling.** Those
kernels are doing everything the hardware will let them do; there is nothing to
tune inside them. The work itself is the problem: a `(B,C,T,3)` float tensor is
**868 MB** at N=4096, and the encoder builds several of them
(`rel`, `local`, `force_local`, `weights*force_local`) plus a dozen `(B,C,T)`
fields at 289 MB each. Peak torch allocation: **4.045 GB**.

*Evidence:* per-op achieved-bandwidth column above (every elementwise row
between 194 and 246 GB/s); AI of 0.192 vs a 70.7 ridge; and §4's fusion
experiment, which removes only the round trips and recovers 2.40×.

### 2. `einsum` lowered to batched tiny GEMMs — 37.81 ms (34.8%), bound by neither wall

`torch.einsum("btji,bctj->bcti", site_mat, rel)` becomes: permute `rel` to
`(B,T,C,3)` → **`aten::copy_`, 7.63 ms, pure data movement** → `aten::bmm` of
`[1507328, 48, 3] × [1507328, 3, 3]` → **24.53 ms**. nsys shows that single
`bmm` fanning out into **6 launches of `cutlass_80_simt_sgemm_64x64_8x5_nn_align1`**
— a kernel tiled 64×64×8 being handed M=48, N=3, K=3. It reaches **73.0 GB/s
(28% of the DRAM ceiling) and 25 GFLOP/s (0.14% of the compute ceiling)**
at the same time.

That double miss is the signature of a launch-configuration/tile-shape problem
rather than a resource limit, and it is the one place where "the shapes are fine
but the kernel isn't" is true. A 3×3 rotation applied per element is not a GEMM
and should never have been dispatched as one. The second einsum
(`force_local`, `[4096,48,3] × [4096,3,1104]`) is better behaved at 198 GB/s but
is still doing something absurd: it expands `(B,C,3)` + `(B,T,9)` — about 20 MB
of real input — into **868 MB** of output, and every byte of that is read back
one op later and multiplied by a weight that is zero 99% of the time.

*Evidence:* op table rows; `nsys` kernel summary
(`profiling/nsight/enc_N4096_kernsum.txt`); the simultaneous 28%-of-memory /
0.14%-of-compute reading.

### 3. Computing 72.4M (env, contact, taxel) triples when ~0.5M carry signal

The encoder evaluates the full dense product. Two independent masks then throw
almost all of it away *after* it has been computed and written to DRAM:

| | pairs at N=4096 | fraction |
|---|---|---|
| dense `B × C × T` | 72,351,744 | 100% |
| live contacts only (measured **3.92/env**, budget C=48) | 5,907,251 | 8.2% |
| …and only the touched body's taxels (368/12 ≈ 31) | **492,271** | **0.68%** |

**99.3% of the arithmetic and its traffic is multiplied by zero** at
`weights = weights * keep`. This is not a subtle inefficiency, it is the
dominant one — and unlike #1 and #2 it cannot be fixed by fusion, only by not
visiting the pairs. Note the interaction with #1: it means the traffic in #1 is
not merely avoidable in principle, it is avoidable *by 145×* in practice.

*Evidence:* `--mode real` measured 4014 live contacts across 1024 envs
(3.92/env) against a 48-slot budget; `taxel_body` puts 368 taxels on 12 bodies
and `body_mask` restricts each contact to exactly one of them; `dropped_contacts
= 0` at every N in the sweep, so the 48-slot budget is ~11× oversized for this
scene and the padding is pure waste.

### Explicitly refuted

- **Kernel-launch bound: NO.** 58 launches × 3.34 µs = 0.194 ms = 0.18%.
- **Compute bound: NO.** 0.19% of the measured SGEMM ceiling.
- **Occupancy/latency bound: NO, for the 59% of time in elementwise kernels** —
  they hit 87% of the DRAM ceiling, which is not something a latency-starved
  kernel does. **YES for the `local` bmm specifically** (cost #2), which is
  neither bandwidth- nor compute-limited and is therefore limited by its own
  shape. I could not confirm this with an occupancy counter because `ncu` is
  blocked; it is inferred from the simultaneous double miss.

---

## 4. The controlled experiment

`--mode fusion` compiles the *unmodified* `encode_taxels` with
`torch.compile` (inductor). This changes no FLOP, no algorithm, no sparsity, and
no occupancy strategy. It removes DRAM round trips and nothing else. Inductor
cannot fuse the two einsums, so it still materialises `local` and `force_local`
— this is a **partial** fusion and therefore a **lower bound** on what a
hand-written kernel gets.

| N | eager | fused | speedup | launches | max abs err |
|---|---|---|---|---|---|
| 1024 | 27.35 ms | 11.22 ms | **2.44×** | 37 → 13 | 4.8e-7 |
| 4096 | 109.20 ms | 45.47 ms | **2.40×** | 58 → 34 | 4.8e-7 |

A 2.4× speedup from removing memory traffic alone, at identical arithmetic and
identical numerics, is the cleanest available proof that the encoder is
traffic-bound. If it were compute-bound this would have been 1.0×; if it were
launch-bound the 24 launches removed would have bought 0.08 ms, not 63.7 ms.

**Side finding worth acting on:** `torch.compile` on the existing encoder is a
free **1.82× end-to-end step speedup** at N=4096 (145.6 → 80.1 ms), available
today with a one-line change and no new kernel. It is not a substitute for the
kernel work — it leaves 45 ms on the table against a 0.43 ms traffic floor — but
it is a sensible safety net to have banked before writing Triton, and it is a
useful second baseline to measure the hand kernel against.

---

## 5. Verdict for the kernel design

### The floor any design is aiming at

A kernel that keeps intermediates in registers moves only inputs and outputs:

| | N=4096 |
|---|---|
| inputs (`site_pos`, `site_mat`, contact arrays) | 85.7 MB |
| output `(B, 368, 3)` | 18.1 MB |
| **total compulsory traffic** | **103.8 MB** |
| at the measured 240 GB/s copy rate | **0.43 ms** |
| current traffic | 19,994 MB (**193× more**) |

### Which option the evidence supports

**(a) two-pass gather, recomputing weights — SUPPORTED, and it is the right
call. But not for the reason H6 gives.**

- The two-pass structure is **forced, not chosen**. `weights /= weights.sum()`
  normalises per contact over the taxel axis. In a gather kernel, the taxel axis
  is the *parallel* axis, so the normaliser cannot exist in the same pass that
  consumes it. Pass 1 must reduce over taxels per (env, contact); pass 2
  gathers. This is not a design preference; any correct formulation needs it.
- **Recompute beats store because traffic costs 368× what FLOPs cost here.**
  In a dense layout, storing `weights` is 289 MB written + 289 MB read at
  N=4096 = **2.4 ms at the ceiling**; recomputing the weight is ~20 FLOP per
  visited pair, which at 0.5M visited pairs is 10 MFLOP ≈ **microseconds**. Not
  a close call.
  *Honest caveat:* in a fully body-sparse layout the stored-weight buffer is
  only ~0.5M floats (2 MB), so option (b)'s traffic penalty largely evaporates
  and the two become close on speed. At that point (a) still wins, but on
  simplicity and footprint — no second buffer, no index structure, no
  O(B·C·T) allocation — rather than on bandwidth. Say that, rather than
  claiming a bandwidth win that the sparse case does not support.

**(c) one-pass scatter with atomics — REFUTED, and refuted twice.**
It cannot be one-pass at all: the normaliser is a reduction over exactly the
axis it would scatter into, so it needs the same pass-1 as (a). Having paid that
cost it then buys nothing — the traffic is identical to (a) — while adding
atomic contention (contacts on one body all target the same ~31 taxels) and
run-to-run non-determinism. Strictly dominated.

**(d) what the data adds on top:** the gather/scatter axis is **not where the
speed is**. Costs #1 and #3 say the wins are (i) never materialising
`(B,C,T,·)`, and (ii) never visiting the 99.3% of pairs the masks zero. Gather
is simply the formulation that makes both natural — one block per (env, body
segment), looping that env's live contacts, accumulating in registers. So the
concrete design recommendation is:

> **Two-pass gather over `(env, body-segment)`, recomputing weights, iterating
> only the live contacts of that env whose `contact_body` matches the segment.**

The three things to get right, in order of measured payoff: (1) nothing bigger
than the output ever reaches DRAM; (2) loop live contacts, not all 48 slots
(≈11×); (3) loop the body's ~31 taxels, not all 368 (≈12×). Note that (2) needs
a per-world contact count on the device — the harness already computes the
bucketing that would give it, and reads it at ~0.65 ms/step, so this is cheap.

**Also worth registering against H6:** the determinism half of H6 is supported
*by construction* — a gather accumulates in a fixed slot order in registers, so
it is reproducible — but nothing here **measured** it, because the current
harness is already deterministic (it sorts rather than using atomics). H6's
determinism claim remains untested, not confirmed. And H6's implied mechanism —
that scatter is slow because of atomics — is **not** what the profiler found.
Atomics never appear. The encoder is slow because it writes 20 GB to DRAM per
step, 99.3% of it to be multiplied by zero.

### Estimated speedup, and the assumptions behind it

**Encoder: 20–40× is the defensible projection; ~100× is the optimistic bound.**

Assumptions, stated so they can be checked when the kernel exists:

1. Traffic drops to the 103.8 MB floor → 0.43 ms at 240 GB/s. A real gather
   kernel will not be perfectly coalesced on the contact reads; assume 50%
   efficiency → **~0.9 ms**.
2. Arithmetic: with live-contact + body-segment sparsity, ~0.5M pairs × ~53
   FLOP ≈ 26 MFLOP — negligible. Even the *dense-over-48-slots* variant (3.84
   GFLOP) costs only **~1.9 ms** at a conservative 2 TFLOP/s for a kernel with
   `exp` in the inner loop.
3. So the kernel lands at **~1–3 ms**, i.e. **35–100×** against 108.75 ms. I am
   quoting **20–40×** as the number to plan against, because it discounts for
   the things this analysis cannot see: register pressure forcing spills, the
   load imbalance from envs with unequal contact counts, and the tail effect of
   only 48 SMs.
4. This assumes the physics half is untouched and the contact bucketing stays at
   its measured 0.65 ms/step at N=1024.

**Amdahl bound — the number that actually constrains the project.**

The encoder is **77.1% of step time at N=4096**, so **even an infinitely fast
encoder caps total step speedup at 1/(1−0.771) = 4.37×.**

| encoder speedup | step time | total speedup | env-steps/s |
|---|---|---|---|
| 1× (today) | 145.6 ms | 1.00× | 28,177 |
| 2.4× (`torch.compile`, free) | 80.1 ms | 1.82× | 51,210 |
| 10× | 44.6 ms | 3.27× | 92,069 |
| **20×** | 38.9 ms | **3.74×** | 105,339 |
| **30×** | 37.1 ms | **3.93×** | 110,655 |
| 100× | 34.5 ms | 4.23× | 119,068 |
| ∞ | 33.3 ms | **4.37×** | 123,078 |

**Read the diminishing returns carefully.** Going from a 30× kernel to a 100×
kernel buys 3.93× → 4.23×, i.e. **7.6% more end-to-end throughput for 3.3× more
kernel performance.** The project's headline number is essentially saturated at
a 20–30× encoder. Past that point the honest next move is to profile the physics
step, not to keep tuning the taxel kernel. Worth deciding that now, in advance,
rather than after three more days of kernel tuning.

For context on the ceiling: at 3.93× the tactile path reaches 110,655
env-steps/s against a **physics-only** 129,682 — i.e. touch would cost a 1.17×
throughput penalty instead of today's 4.60×.

**Secondary win, not on the critical path but real:** peak torch memory at
N=4096 drops from **4.045 GB to ~0.10 GB**, a 39× reduction. On a box where an
oversized allocation wedges the host, and with H5 open on whether memory or time
is the binding constraint at high N, that is worth banking.

---

## Reproducing

Every command must be memory-capped — this box has ~121 GB shared between CPU
and GPU and an oversized run takes down the host, not the process.

```bash
CAP="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0"
PY=/home/afthabshiraz/Mujoco/.venv/bin/python

$CAP $PY profiling/profile_encoder.py --mode bandwidth              # the two ceilings
$CAP $PY profiling/profile_encoder.py --mode bench --num-envs 1024 --bandwidth
$CAP $PY profiling/profile_encoder.py --mode bench --num-envs 4096 --bandwidth \
         --iters 10 --profile-calls 3
$CAP $PY profiling/profile_encoder.py --mode fusion --num-envs 4096 --iters 10
$CAP $PY profiling/profile_encoder.py --mode real  --num-envs 1024   # vs real contacts

# nsys (ncu counters are unavailable on this box -- see §0)
$CAP /usr/local/cuda/bin/nsys profile --capture-range=cudaProfilerApi \
    --capture-range-end=stop -t cuda -o profiling/nsight/enc_N1024 \
    $PY profiling/profile_encoder.py --mode ncu --num-envs 1024
```

Do not exceed 4096 envs. N=4096 allocates 4.045 GB of torch tensors in the eager
path; the fusion path allocates less but compiles for ~30 s first.
