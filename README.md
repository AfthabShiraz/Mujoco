# Tactile observations at training scale

**What does touch cost when you simulate thousands of hands at once, and can you
make it cost nothing?**

This repository prices **four** ways of obtaining a tactile observation inside a
GPU-parallel MuJoCo Warp simulation of the LeapXELA hand, locates the env count
at which the tactile encoder — not the physics — becomes the bottleneck, and then
removes it with a hand-written Triton kernel.

**Headline result:** on an NVIDIA DGX Spark (GB10), at the supervisor's own
`sim_dt = 0.01`, a faithful batched port of the 368-taxel Gaussian-splat encoder
reaches **76.9% of step time at N = 4096**, where turning touch on costs a
**4.57×** throughput penalty. Because the encoder is 76.9% of the step, removing
it entirely would gain at most **4.34×** end to end — the Amdahl bound, stated
before any optimisation was attempted. A two-pass Triton gather kernel then makes
the encoder **188× faster than eager torch and 78× faster than `torch.compile`**
(109.5 ms → 45.5 ms → 0.582 ms per call at N = 4096), which against that **4.34×
ceiling** delivers **4.18× end to end** — 96% of everything that was available.
Throughput goes 28,006 → **116,961 env-steps/s** against a **128,061**
physics-only ceiling, so **the cost of touch falls from 4.57× to 1.09×** — and to
**1.01×** under scripted grasp motion.

**Second result, not specific to tactile work:** mujoco_warp dispatches over the
**allocated** contact budget, not the live contact count, so step time is affine
in `nconmax` at fixed occupancy — `ms/step ≈ 5.3 + 0.145 × nconmax` at N = 1024.
**`nconmax` is a throughput parameter worth 3.4×, not a safety margin.**

**The full write-up is `report.tex`.** This README is the repository guide; the
report is the paper.

![Touch is now nearly free](analysis/plots/E_touch_is_free.png)

---

## 1. The gap

The supervisor's GPU training stack (`leapXelaMjLab`, mjlab + MuJoCo Warp +
multi-GPU PPO) trains cube reorientation on a *tactile* hand at 4096
environments, and its observation set — joint positions and velocities, cube pose
error, cube velocities, fingertip positions — contains no tactile term of any
kind. The same lab's `leapXELA_model` holds the reference 368-taxel
Gaussian-splat encoder (`VirtualTaxelSensor`) — the correctness standard used
throughout this repo — but it is a per-contact Python loop that runs on CPU,
against a *different* hand model lineage. That hardware-faithful splat
encoder — 3 channels per taxel, no added collision geoms, taxel indices equal to
real XELA hardware IDs — does not exist on GPU anywhere, and nobody has measured
what it costs at training scale.

**Not overclaimed:** native MuJoCo `<touch>` sensors *do* already run on GPU —
`mujoco_warp/_src/sensor.py` implements `_sensor_touch`, dispatched over
`(naconmax, n_touch_sensors)`. What is missing is (a) the splat representation on
GPU and (b) the cost measurement for any representation at scale. Prior art
`langxin11/contactile_mjlab` puts tactile observations in mjlab, but for a
Robotiq 2F-85 gripper with 18 built-in `<force>` sensors on added sphere geoms —
a different robot, a different representation, and no scaling study.

## 2. Status

**H1, H2, H3, H4 and H6 are resolved** (`HYPOTHESES.md`); **H5** (unified memory
moves the ceiling) is the one still open, and the evidence so far runs against
it — memory has failed to bind in every measurement taken. All four
representations are priced, the kernel is built and validated, and the encoder
runs inside the supervisor's own training environment.

Everything below is measured on this hardware, for this scene, at
`sim_dt = 0.01` to match `leapXelaMjLab/env_cfg.py`, and is reported as such.
Earlier sweeps taken at the scene's default `dt = 0.001` are retained in
`benchmarks/results/` without the `_dt01` suffix and are **superseded** — their
ratios were always sound, but only the `_dt01` numbers are comparable to his
training throughput.

Not done: nothing has been priced with a policy, an optimiser and rollout
buffers attached. See §6.

## 3. What was built

| Component | File | What it is |
|---|---|---|
| **Tactile→RL body map** | `explore/taxel_map.py` | The two model lineages share **zero** body names. `BODY_MAP` is derived from the kinematic tree, disambiguated by the layout's own hardware patch names, and validated by body-local mesh bounding box: **8 of 12 bodies match to 0.000 mm**. Chain order is reversed from what the names suggest — `finger` is the *ring* finger — so mapping by name order silently swaps index and ring. |
| **Taxel injection** | `explore/03_inject_taxels.py`, `benchmarks/harness.py:build_scene_model` | 368 sites added to the pinned rigid mjlab model through **`MjSpec`** (67 → 435 sites). mjlab builds its hand through `MjSpec` too, so this transfers into an mjlab `spec_fn` nearly verbatim. Burial was measured by per-body ray cast: max 5.00 mm against the encoder's 10 mm patch-locality gate, so every taxel can still fire. |
| **Oracle fixture** | `explore/04_run_encoder.py` → `explore/out/oracle_fixture.npz` | 30 frames through a scripted grasp, storing the reference encoder's **inputs** (contact pos/frame/geoms/6-vector force, site poses, qpos) beside its 368×3 outputs. Any later implementation is validated without re-running MuJoCo, which sidesteps the Warp-vs-CPU contact discrepancy for pure encoder validation. Includes 8 `ncon = 0` frames. |
| **Widened oracle** | `explore/05_grasp_motions.py` → `explore/out/oracle_fixture_v2.npz` | The seven grasp patterns from his `sparsh-skin-sim/util/motion_util.py`, constants copied verbatim. **632 frames over 56 trials**, taking the taxels that ever fire from 37 to **176 of 368** and fingertip coverage from 2 to **57 of 136**. Both implementations still validate; widening it is what exposed F1–F3 in `HYPOTHESES.md`. |
| **Benchmark harness** | `benchmarks/harness.py`, `benchmarks/scale_sweep.py` | One env count per memory-capped child process, because this box has unified CPU/GPU memory and an oversized run wedges the host rather than raising `CUDA out of memory`. Warmup, `wp.synchronize()` on both sides of the clock, contact counts and overflow reported every run. Two timing passes: fused (headline throughput) and stage-synchronised (the physics/encoder split). `--encoder {torch,compile,triton}` selects the implementation; everything else is held fixed. |
| **Batched torch encoder** | `src/encoders/taxel_torch.py` | Faithful batched port of `VirtualTaxelSensor.update`: dense `(B, C, T)` tensors, no fused kernels, no sparsity. Deliberately unoptimised — its cost is the baseline every later implementation is measured against. |
| **GPU contact plumbing** | `benchmarks/harness.py:SplatEncoder` | Warp keeps contacts in one flat `naconmax` array tagged by `worldid`, unordered; the encoder wants a dense per-env batch. Bucketing is done with a **stable sort**, not atomic counters — slot order then fixes the summation order, so the output is reproducible rather than wobbling at the 1e-7 level. |
| **Profiling harness** | `profiling/profile_encoder.py`, `profiling/RESULTS.md` | Per-op CUDA table joined against a hand-derived byte/FLOP cost model, plus *measured* DRAM and SGEMM ceilings for this machine, plus a controlled `torch.compile` experiment. It is what told the kernel what to do. |
| **Triton kernel** | `src/kernels/triton/taxel_triton.py` | Two-pass gather, recomputing weights. Drop-in replacement for `encode_taxels`, same signature and same semantics. Nothing of shape `(B, C, T, ·)` is ever created. |
| **The other three representations** | `benchmarks/harness.py` (`--tactile {touch,touchgrid}`), `benchmarks/flex_probe.py` | Native `<touch>` (421 sensors) and the 300-geom binary touchgrid are **injected into the same scene**, so all three rigid representations are directly comparable. Native touch has no separable stage to clock — Warp computes it inside `mjw.step` — so its cost is a difference of fused loops against a measured noise floor. Flex is a different body tree and gets its own probe, reported as per-step latency only. |
| **Grasp-motion driver** | `benchmarks/harness.py:GraspDriver` (`--motion grasp`) | Every sweep re-run under his seven patterns instead of one frozen pose. World `w` runs pattern `w % 7` at a phase offset, so the batch spans idle and loaded worlds at once (0–25 contacts/env at N = 4096). `ctrl` comes from a precomputed GPU table — no host round trip in the clock — and is charged to every variant equally. `--motion static` reproduces the old operating point exactly. |
| **The mjlab observation term** | `src/mjlab_tactile/taxel_term.py` | The encoder as an `ObservationTermCfg` inside `leapXelaMjLab` itself: 368 sites injected by wrapping his own `get_spec`, actor obs 57 → **1161** dims. Three silent-or-fatal traps handled: mjlab's `robot/` namespacing, frozen `TaxelEntry` dataclasses, and `wp.set_stream` segfaulting under mjlab's captured CUDA graphs. |
| **The write-up** | `report.tex` | Full IEEE-format report: four representations, profiling, kernel, mjlab integration, threats to validity. |

Two failure modes are worth recording because both produce plausible-looking
output rather than an error:

- `VirtualTaxelSensor` groups taxels by `mj_name2id(entry.body)`. With
  tactile-lineage names against the RL model, every lookup returns −1, all 368
  taxels collapse into one bogus group, and **the encoder returns all zeros
  without raising**. Fixed by remapping the *layout* (`remap_layout`), leaving
  the reference encoder byte-for-byte intact because it is the oracle.
- `njmax` is **per world** in `mjw.put_data` while `naconmax` is a **total**.
  Scaling `njmax` by N made the dense solver O(N²) and throughput *fell* with
  scale: N = 512 went 337 → 67,840 env-steps/s on fixing it.

## 4. Results

Scene: `scene_mjx_cube.xml` from the pinned mjlab assets, hand closed to 60% of
its control range and settled 250 steps so contacts are live (~7.5 contacts/env
at `sim_dt = 0.01`), `nconmax = 48/env`, `njmax = 120/env`, 100 timed steps after
30 warmup steps. **Zero contact-budget overflow and zero dropped contacts at
every N, in all three splat sweeps.**

Throughputs are **physics** steps/s. His config uses `ctrl_dt = 0.05` with
`decimation = 5`, so divide by 5 for control steps: 128,061 physics steps/s at
N = 4096 is 25,612 control steps/s. Every ratio below is unaffected.

### 4.1 H1 — physics dominates when tactile is off → scaling half **confirmed**

`benchmarks/results/scale_sweep_physics_only_dt01.csv`:

| N | env-steps/s | µs/env-step | ms/step | peak GB |
|---|---|---|---|---|
| 1 | 308 | 3246.2 | 3.25 | 1.27 |
| 16 | 3,935 | 254.2 | 4.07 | 1.27 |
| 128 | 31,597 | 31.6 | 4.05 | 1.27 |
| 256 | 38,310 | 26.1 | 6.68 | 1.27 |
| 1024 | 78,349 | 12.8 | 13.07 | 1.27 |
| 4096 | **128,061** | **7.8** | 31.98 | **1.37** |

Scaling is linear to N = 128 — `ms/step` is flat at ~4.05 ms while throughput
doubles at every rung, i.e. 128 environments cost the same wall-clock as one.
Past that the GPU saturates and `ms/step` climbs, though throughput still improves
sublinearly to 4096. Per-env cost falls **416×** from N = 1 to N = 4096.

This also **refuted a scope decision**: D7 assumed the Spark's unified memory
would set the env ceiling. 4096 physics environments use 1.37 GB peak RSS. Memory
was nowhere near binding, and all 13 rungs completed with no failures. The host
crash that motivated D7 was real but was not caused by physics memory at 4096;
its cause is still unidentified, so the memory cap and subprocess isolation stay.

![Throughput vs env count](analysis/plots/B_throughput_scaling.png)

### 4.2 H2 — a naive splat encoder overtakes physics in N ∈ [64, 1024] → **confirmed at N = 256**

`benchmarks/results/scale_sweep_splat_dt01.csv` against the physics-only sweep,
identical scene and settings:

| N | physics-only/s | splat/s | slowdown | physics ms | encoder ms | encoder % |
|---|---|---|---|---|---|---|
| 1 | 308 | 223 | 1.38× | 3.84 | 0.60 | 13.5% |
| 16 | 3,935 | 3,237 | 1.22× | 4.18 | 0.76 | 15.4% |
| 64 | 15,999 | 11,264 | 1.42× | 4.11 | 1.57 | 27.7% |
| 128 | 31,597 | 12,267 | 2.58× | 6.65 | 3.81 | 36.4% |
| **256** | 38,310 | 18,486 | 2.07× | 6.53 | **7.45** | **53.3%** ← crossover |
| 512 | 59,280 | 22,125 | 2.68× | 8.66 | 14.52 | 62.6% |
| 1024 | 78,349 | 25,148 | 3.12× | 12.15 | 28.59 | 70.2% |
| 4096 | 128,061 | 28,006 | **4.57×** | 33.61 | **112.19** | **76.9%** |

![The bottleneck migrates](analysis/plots/A_bottleneck_migration.png)

The mechanism is straightforward. Physics amortises with scale — between N = 1
and N = 4096 it does 4096× the work for **8.8×** the time (3.84 → 33.61 ms) —
while the dense `B × C × T` encoder does not: every environment × 48 contact
slots × 368 taxels, work proportional to N with no reuse, 0.60 → 112.19 ms, a
factor of **187**.

![Encoder share of step time](analysis/plots/C_encoder_share.png)

![Slowdown factor](analysis/plots/D_slowdown_factor.png)

### 4.3 The Amdahl ceiling — 4.34×, fixed before any kernel existed

At N = 4096 the encoder is 76.9% of step time. Eliminating it **entirely** —
making tactile encoding free — therefore caps end-to-end speedup at

> 1 / (1 − 0.769) = **4.34×**

This was written down before any kernel existed, because it is the number every
later result has to be reported against. A kernel that makes the encoder 10×
faster does not make the step 10× faster: it leaves the physics plus a tenth of
the encoder, i.e. **3.3×** end to end. The same arithmetic says a 20× kernel
gives 3.73×, a 30× kernel gives 3.92×, and a 100× kernel gives 4.21× — so going
30× → 100× buys **7.4% more end-to-end throughput for 3.3× more kernel
performance**. That was recorded as a stopping rule, in advance, and it is what
§4.7 concludes with.

Memory never bound the tactile path either: torch peak 4.06 GB and process peak
1.72 GB at N = 4096.

### 4.4 Why the encoder was slow — the profiling verdict

Full report: `profiling/RESULTS.md`. The short version, because it dictated the
kernel design rather than merely preceding it:

- **Bandwidth-bound, decisively.** Arithmetic intensity **0.192 FLOP/byte**
  against a *measured* ridge point of **70.7 FLOP/byte** for this GB10 — 368× to
  the left. It runs at **183.9 GB/s**, 71% of the measured 260.8 GB/s DRAM
  ceiling, and **0.19%** of the measured 18.44 TFLOP/s SGEMM ceiling.
- **The traffic is 19.99 GB per call at N = 4096, and 99.3% of it is multiplied
  by zero after having been written to DRAM and read back.** 72.4M
  (env, contact, taxel) triples are computed; ~492k carry signal. Two masks throw
  the rest away: ~3.9 live contacts against a 48-slot budget, then a body mask
  that keeps 1/12 of the taxels.
- **Launch overhead is 0.18% of the call** (58 launches × 3.34 µs). Any design
  justified on "fewer kernel launches" is justified on the wrong grounds — so the
  kernel below does *not* try to be a single kernel.
- **The controlled experiment.** `torch.compile` on the *unmodified* encoder
  changes zero FLOPs, zero algorithm and zero occupancy strategy; it removes only
  DRAM round trips, and it is **2.41× faster** at N = 4096, numerically identical
  to 4.8e-07. That is the cleanest available proof that traffic is the limiter,
  and it also fixes the honest baseline: **2.41× is available from one line of
  code**, worth 1.82× end to end, so the Triton kernel is reported against
  `torch.compile` as well as against eager.
- **H6's stated mechanism was refuted.** H6 predicted the win would come from
  replacing a scatter's atomic contention with a gather. **Atomics never appear
  in the profile.** The gather is still the right formulation, but because it is
  the one that makes it natural never to materialise `(B, C, T, ·)` and never to
  visit the 99.3% — not because of contention.
- The compulsory traffic floor — inputs plus output, nothing else — is
  **103.8 MB**, i.e. 193× less than what the dense encoder moves.

### 4.5 The kernel

`src/kernels/triton/taxel_triton.py`. Two-pass gather, recomputing weights rather
than storing them.

The two-pass split is **forced, not preferred**: `weights /= weights.sum()`
reduces over the taxel axis, and the taxel axis is pass 2's parallel axis, so
pass 2 cannot compute its own denominator. The same argument independently kills
the one-pass scatter alternative.

- **Pass 1** — one program per `(env, contact)`. Walks that contact's body's
  taxels, computes each Gaussian weight exactly as the reference does, writes a
  single float `weight_sum[B, C]`. No collisions, no atomics.
- **Pass 2** — one program per `(env, taxel tile)`. Loads the tile's site pose
  once, loops the env's contact slots, recomputes the weight, divides by
  `weight_sum`, rotates the contact wrench into the taxel frame, and accumulates
  **in registers**. Three stores at the end.

Recompute beats store because traffic costs ~368× what FLOPs cost here; the only
thing that reaches DRAM between the passes is one float per `(env, contact)` —
786 KB at N = 4096. Tile sizes (`BLOCK_T` = 64 for pass 1, 32 for pass 2) were
chosen by a measured 3×3 sweep, not by taste.

`benchmarks/results/kernel_bench.csv`, encoder only, 48-slot budget, 368 taxels:

| N | eager | `torch.compile` | **Triton** | vs eager | vs compile | GB/s | % of measured ceiling |
|---|---|---|---|---|---|---|---|
| 128 | 2.887 | 1.282 | **0.038** | 74.9× | 33.3× | 110.3 | 42.3% |
| 512 | 13.792 | 5.474 | **0.068** | 203.2× | 80.7× | 250.5 | 96.1% |
| 1024 | 27.340 | 11.196 | **0.132** | 207.5× | 85.0× | 258.1 | 99.0% |
| 2048 | 55.177 | 22.437 | **0.274** | 201.1× | 81.8× | 247.8 | 95.0% |
| 4096 | 109.525 | 45.536 | **0.582** | **188.2×** | **78.3×** | 233.8 | 89.6% |

![Encoder cost, three ways](analysis/plots/F_encoder_three_ways.png)

Modelled traffic at N = 4096 falls **19.99 GB → 136 MB**, against the 103.8 MB
compulsory floor — 1.31× the floor, and 147× less than eager. Peak torch
allocation falls **4.095 → 0.183 GB**. This exceeds the 20–40× projected in
`profiling/RESULTS.md`; the discounts that projection assumed (register spills,
load imbalance, a 48-SM tail) did not materialise.

**Determinism, now measured.** 20 runs at B = 1024 on identical input, compared
with `torch.equal` over 1,130,496 floats: **20/20 bit-identical**. Both passes
accumulate in registers in a fixed order and neither uses atomics.
`profiling/RESULTS.md` had recorded H6's determinism claim as supported by
construction but never measured, because the existing torch path is already
deterministic (it sorts rather than using atomics). `tests/test_triton_kernel.py`
measures it.

### 4.6 End to end — touch is now nearly free

`benchmarks/results/scale_sweep_splat_triton_dt01.csv`, same scene and settings,
the kernel wired into the harness behind `--encoder triton`, correctness re-gated
in-harness at **1.49e-07** before any timing was recorded.

Against the **4.34× Amdahl ceiling** established in §4.3:

| N | no touch | torch splat | **Triton splat** | cost of touch | vs torch |
|---|---|---|---|---|---|
| 64 | 15,999 | 11,264 | 13,960 | 1.15× | 1.24× |
| 256 | 38,310 | 18,486 | 34,715 | 1.10× | 1.88× |
| 1024 | 78,349 | 25,148 | 77,361 | **1.01×** | 3.08× |
| 4096 | 128,061 | 28,006 | **116,961** | **1.09×** | **4.18×** |

**The cost of tactile sensing at 4096 envs falls from 4.57× to 1.09×** — from a
357% penalty to 9%. The end-to-end gain of **4.18×** is **96% of the 4.34× that
was available**. The encoder's share of step time drops from **76.9% to 6.5%**.
In-harness torch peak allocation drops 4.06 → 0.12 GB.

**Under real grasp motion it holds, and improves to 1.01×.** Re-run with
`--motion grasp` (his seven patterns, worlds decorrelated, 0–25 contacts/env at
N = 4096): physics-only 126,637, eager splat 27,642 (**4.58×**, unchanged as its
contact-blind shape predicts), Triton 125,535 (**1.01×**). The kernel *gains*
because pass 2 is indexed by (env, taxel tile) — an idle world costs a scan of
its slot array, not an evaluation, and with no atomics it does not stall a loaded
world either. ⚠ Read 1.01 as *unresolvable*, not as a bound: at N = 1024 the
Triton path measures nominally faster than doing no tactile work at all.

![The migration is undone](analysis/plots/G_migration_undone.png)

The migration H2 found is undone: the Triton tactile stage never overtakes the
physics step at any N in the sweep, peaking at 10.8% of step time.

### 4.7 Amdahl is now the whole story

![Amdahl](analysis/plots/H_amdahl.png)

Plugging the measured kernel speedup into the 76.9% share gives 4.26× predicted;
the sweep measured 4.18×. Those agree, which is the check worth having — the
end-to-end number is not doing anything the stage split does not already explain.

The point of the figure is the flatness on the right. A **188×** kernel and a
hypothetical **30×** kernel land **0.34× apart** end to end (4.26× vs 3.92×),
because both are already deep into the ceiling's shadow. **The remaining ~10% of
end-to-end throughput lives entirely in the physics step.** Further tuning of
this kernel is not worth the days, and that conclusion was pre-registered in
§4.3 rather than reached after the fact.

### 4.8 What this kernel result does *not* claim

Every one of these is a limitation of the measurement, and each is recorded in
`HYPOTHESES.md` H6.

- **Runtime is now data-dependent, and only the upper end is still unmeasured.**
  The dense baseline's cost was independent of how many contacts were live; the
  sparse kernel's is not. Measured at N = 4096: **0.54 ms at 1 live contact/env,
  0.57 ms at ~4.3, 0.73 ms at 12, and 1.27 ms with all 48 slots live.** The
  narrower version of this worry — that the headline came from one frozen pose
  where every world was identical — is **retired** by the `--motion grasp` sweeps
  (§4.6), which span 0–25 contacts/env across the batch and *improve* the result.
  What remains is the top: **no measured condition holds every world near the
  48-slot cap**, and that synthetic case, while still **86× eager / 36×
  `torch.compile`**, is 2.2× slower than the measured regime. A grasp that
  saturates the budget would cost more than 1.01×. The profiling report's
  "synthetic inputs are performance-equivalent to real ones, ratio 1.0006" result
  was established for the dense encoder and **does not carry over to this
  kernel**.
- **Read the 96–99%-of-ceiling figures at N = 512–2048 with suspicion.** Every
  achieved-GB/s number in this repository is *modelled bytes ÷ measured time*,
  not a counter read. At mid-N that ratio lands within a few percent of the DRAM
  ceiling, which almost certainly means **L2 reuse is being charged as DRAM
  traffic**, not that the wall was approached. **89.6% at N = 4096 is the
  defensible figure**; the mid-N numbers are reported because deleting them would
  be worse, not because they should be believed.
- **There is no Nsight-reported limiter anywhere in this work.** `ncu` returns
  `ERR_NVGPUCTRPERM` on every section, including `LaunchStats` and `Occupancy`:
  the driver is built with `NVreg_RestrictProfilingToAdminUsers=1` and there is
  no passwordless root on this box. Substituted throughout: an empirically
  measured saturating-copy bandwidth ceiling (260.8 GB/s read, 240.7 copy, 237.6
  triad), an SGEMM compute ceiling (18.44 TFLOP/s, TF32 off), and modelled bytes
  ÷ measured time. `nsys` does work and corroborates the kernel-level picture.
  The `torch.compile` experiment in §4.4 exists precisely because it tests the
  same conclusion without needing counters.
- **The bottleneck has migrated a second time, one level down, and the split is
  inferred rather than instrumented.** At N = 4096 the harness reports **2.393 ms
  for the tactile stage**, while the kernel alone measures **0.582 ms**. The
  remaining ~1.8 ms is the stage's pre-processing: fetching contact forces from
  Warp (`mjw.contact_force`) and scattering the flat `naconmax` contact array into
  the padded per-env `(B, C, ·)` layout. That makes pre-processing **~75% of the
  tactile stage**. **This split is inferred from two separately measured numbers
  and has not been directly instrumented** — the two were taken on different input
  paths (in-harness vs. the encoder-only bench), so it should be measured directly
  before being quoted anywhere else. It is this project's own thesis recurring one
  level down, and it is the cheapest remaining target.
- **On recompute vs. store, honestly:** in a fully body-sparse layout the stored
  weight buffer would be only ~2 MB, so store's bandwidth penalty largely
  evaporates and the two designs come close on speed. Recompute still wins there,
  but on simplicity and footprint — no second buffer, no index structure, no
  `O(B·C·T)` allocation — rather than on bandwidth.

### 4.9 The other three representations — the comparison that makes the kernel mean something

A kernel that removes a 4.57× overhead is a speedup in a vacuum. Priced against
the alternatives, it is the difference between the hardware-faithful
representation being unaffordable and being about as cheap as the built-in one.
All at N = 4096, `sim_dt = 0.01`, frozen pose:

| Representation | Outputs | `nconmax` | env-steps/s | cost |
|---|---|---|---|---|
| None (physics only) | — | 48 | 128,061 | 1.00× |
| Native `<touch>` | 421 | 48 | 125,877 | **1.02×** |
| Splat, Triton | 1104 | 48 | 116,961 | **1.09×** |
| Splat, dense torch | 1104 | 48 | 28,006 | 4.57× |
| Binary touchgrid | 300 | **256** | 29,440 | 4.35× |
| Flex skin † | — | 48 | — | **~540×** |

† Different body tree, not injectable into the RL model, so no rate is given —
`env_steps_per_sec` is **not** a like-for-like bar. The ratio is at **N = 256**
and at **matched simulated time**: flex's per-step latency is 107.6× the rigid
hand's, and its 0.002 s timestep advances one fifth as far as the 0.01 s used
everywhere else, so 107.6 × 5 ≈ 540. Quoting raw steps/s would flatter flex 5×.

**Native `<touch>` is essentially free and already works on GPU** (H3 context).
Warp computes it inside `mjw.step`, so its cost is a difference of fused loops:
0.308 ms against a ±0.047 ms noise floor at N = 4096, ~1% of the step. **Tell
Hamid this one** — if a scalar-per-zone readout suffices for a task, it is
available today at a price that is hard to measure.

**The touchgrid's cost is not where anyone predicted (H3, and the most
generalisable result here).** It cannot run at his `nconmax = 48` at all — it
dies with `Warp CUDA error 700: illegal memory access` in `ccd_kernel`, needing
ncon 232 / nefc 944 against the bare scene's 24 / 112. But the contacts are not
what costs. Holding the budget fixed and turning the patches' collisions on costs
**nothing**; the entire 4.3× collapse sits between two rows that differ only in
how much contact space was *allocated*:

| variant | geoms | nconmax/njmax | env-steps/s | vs bare |
|---|---|---|---|---|
| physics only | 71 | 48 / 120 | 128,061 | — |
| touchgrid, collide off | 371 | 48 / 120 | 124,963 | −2.4% |
| touchgrid, collide off | 371 | **256** / 2048 | 28,926 | −77.4% |
| touchgrid, cube only | 371 | **256** / 2048 | 29,440 | −77.0% |

`njmax` costs nothing; `nconmax` is the parameter. `benchmarks/contact_budget_sweep.py`
scans it at fixed N with occupancy pinned at **7.4–7.7 contacts/env at every
point**, and runs the scan on two scenes 5× apart in geom count:

```
bare scene (71 geoms)         ms/step = 5.51 + 0.1428 x nconmax   R² = 0.998
grid, collide off (371 geoms) ms/step = 4.73 + 0.1436 x nconmax   R² = 0.996
```

**The two slopes agree to 0.6%** — the cost attaches to the allocation, not to
the geometry and not to the occupancy. Throughput falls 78,045 → 17,062
env-steps/s across the scan (4.6×) with identical physics. The mechanism is in
the source: mujoco_warp sizes its launches off `d.naconmax`, the allocation, not
the live count — most expensively `smooth.py`'s `dim=(nacttrnbody, naconmax, nv)`.
**Every step pays for every slot.**

Under grasp motion the peak per-world contact count rises 232 → **390**, so the
budget a grid needs is a property of the *motion*, not of the pose you benchmark.

**Flex runs on Warp — and is ~540× too slow.** `put_model` accepts it, which
MJX-JAX refuses outright (`NotImplementedError`). At its own required
`timestep = 0.002` (it goes unstable at 0.01), N = 256 delivers 71 equivalent
env-steps/s against the rigid hand's 38,310, and it has already stopped scaling.
Ablation: ~63% forward dynamics over 387 bodies / 1120 DoF, ~37% constraint
solver, **~1% contacts**. No kernel can help — it is all inside `mjw.step`.
**The predicted failure mode was wrong**: memory never bound (1.22 GB at
N = 1024), time did.

### 4.10 Inside his actual training environment

The figures above price a bare `mjw.step` loop. `src/mjlab_tactile/taxel_term.py`
puts the encoder inside `leapXelaMjLab` itself — his task, his rewards, resets and
randomisation — as an `ObservationTermCfg`. Actor observation 57 → **1161** dims,
1104 of them live taxel readings.

At N = 2048: **14.05 ms without tactile, 16.67 ms with 368 taxels — 1.19×**,
against the harness's 1.17× prediction at the same N. **The isolated benchmark
transfers.**

Getting there required fixing an unrelated mjlab bug worth **3.31×**:
`EntityData.site_pos_w` gathers every site's position *and* orientation, runs
`quat_from_matrix` over all of them, then slices out three columns and discards
the quaternions. Harmless at 68 sites; the largest cost in the environment at
436. It speeds up the *no-tactile* env too (16.16 → 14.05 ms), which is the proof
it is not our bug. Substitution verified bitwise. See `contrib/mjlab-site-pos/`.

## 5. Correctness

Timing is only worth reporting if the answer is right, and the documented failure
mode here is a *fast wrong answer* (all-zero taxels, or forces attributed to the
wrong finger). Every check below ran before the timing it accompanies.

| Check | What it compares | Result |
|---|---|---|
| Batched torch encoder vs CPU reference, **float64** | `tests/test_encoder_oracle.py`, v2 fixture, all 632 frames | **4.16e-07 max abs error** — the real semantic-equivalence claim |
| Same, in float32 | v2 fixture | **4.59e-02 max relative** on loaded palm taxels — *not* a bug; see the gate note below |
| Batching vs per-frame | frames padded into one batch vs one at a time | **bit-identical** (0.0e+00) — the padding and masking are inert |
| GPU torch path vs CPU encoder, identical contacts | `benchmarks/harness.py --verify`, check A (gate) | **1.5e-08 max abs error** |
| GPU torch path vs `VirtualTaxelSensor` on a CPU re-solve | check B (reported, not gated) | **3.8e-03** — Warp and CPU MuJoCo generate different contact sets for the same state; an engine difference, not an encoder bug |
| **Triton kernel vs the oracle fixture** | `tests/test_triton_kernel.py`, v1 | **1.94e-06 max abs error** |
| **Triton kernel vs the widened fixture** | same test, v2 (632 frames) | **2.14e-04 max abs / 4.59e-02 max rel** — the same float32 gate effect, not a kernel difference |
| **Triton kernel vs the real Warp contact stream** | same test, N = 64 | **1.34e-07** |
| **Triton kernel, re-gated in-harness before the sweep** | `--encoder triton --verify` | **1.49e-07** |
| **Triton kernel determinism** | 20 runs, B = 1024, `torch.equal` over 1,130,496 floats | **20/20 bit-identical** |
| `torch.compile` vs eager | `profiling/profile_encoder.py --mode fusion` | **4.8e-07** — numerically identical, as it must be |

The reference accumulates in float64 and casts to float32 on return; both GPU
implementations run in float32 throughout, so agreement is to float32 rounding
rather than bit-exact. The Triton kernel additionally sums the weights and the
contact accumulation in a different order from the torch version, and float
addition is not associative, so the two are *not* expected to be bit-identical to
each other — 1.94e-06 against the oracle is consistent with float32 rounding on
peak per-channel forces of 5.4 N.

The kernel's edge cases are tested individually rather than assumed, because each
skip it performs is an opportunity to be plausibly wrong: `ncon = 0` (exact
zeros, not NaN from a 0/0 normalisation, and not a launch of an empty grid); a
contact whose `weight_sum` is 0 because it is gated out; padding slots holding
live-looking junk in `contact_pos`/`contact_force`; and a contact on a body
carrying no taxels at all.

**The fixture was widened, and widening it found three things.** v1 was narrow —
30 frames, **37 of 368 taxels ever firing**, 31 of them on the palm and **2 on
fingertips**. v2 (`explore/05_grasp_motions.py`, his seven patterns) is **632
frames over 56 trials**, **176 of 368 taxels**, **57 of 136 fingertip taxels**,
shear range ±5.4 N. Both fixtures are still tested, and all **11 tests pass**.

- **F1 — the encoder's gate is discontinuous, and float32 lands on it.** The
  reference gates each splat with hard thresholds and then renormalises over the
  *survivors*. In this scene the palm taxel plane sits at `|local_z|` = 10.0000 mm
  against a 10 mm cutoff — margins of **0 to 0.2 µm**. float32 cannot resolve
  that, so a taxel falls on the opposite side of the gate from the float64
  reference and rescales the survivors, giving a clean **4.59% relative** error on
  all three channels at once. The same torch code in float64 collapses it to
  **4.16e-07**. **Any float32 GPU port of this encoder will show this**; it is a
  property of the algorithm's gate, not of the kernel. The tests therefore assert
  a tight float64 bound (1e-6, the real claim) plus a documented loose float32 one.
- **F2 — 48 medial taxels are structurally unreachable in this scene.** Every cube
  contact on the medial links is **11.50 mm** out of plane, a constant above the
  10 mm cutoff. **No motion fixes this** — the pads sit on a different face of the
  link from the one the cube presses. Verified as a property of *his layout*, not
  of our transplant: the taxel-to-surface geometry is identical in both lineages.
- **F3 — the fingertip mismatch (D4) is a knife-edge.** Fingertip contacts sit
  **9–10.8 mm** out of plane, straddling the gate, and 33–49% produce nothing.
  This is the RL tip's extra ~10.3 mm of distal length, which carries no taxels.

**Still narrow:** 192 taxels remain dark, and max contacts is 19 against the
48-slot budget — a one-cube scene does not saturate it.

## 6. Known limitations

- **136 of 368 taxels sit on a geometrically different part.** The tactile and RL
  model lineages carry different fingertips: the RL tip is ~10.3 mm longer
  (tactile lineage 30 × 61.98 × 34.6 mm; RL lineage 34.85 × 72.31 × 39.2 mm),
  and identically so across all three mjx variants, so this is not a
  fingertip-variant choice. The mounting frame agrees exactly — faces at `min_x`, `max_y`,
  `min_z` align — so the 232 palm/proximal/middle taxels transfer with no
  transform, but the *lateral* placement of the 136 fingertip taxels is
  unverified against the physical robot. This affects sim-to-real fidelity, not
  throughput or bottleneck location. It is an open question for the supervisor:
  which tip is the real robot, and is there a variant matching the sensorised tip?
- **The benchmark measures physics + encoder, not the full training loop.** It
  drives `mujoco_warp` directly on the mjlab scene; there is no policy, no
  optimiser, no rollout buffer, no logging. Encoder share, the 4.34× ceiling and
  the 4.18× achieved against it are all properties of the *simulation* step;
  adding the RL machinery will lower every one of them, and it will lower the
  headline "touch is nearly free" claim by making the tactile stage a smaller
  slice of a bigger step. §4.10 closes part of this gap — the term now runs
  inside his `ManagerBasedRlEnv` with managers, resets and randomisation — but
  **still with a fixed action tensor**, so no policy or optimiser is attached.
  Note the bound runs both ways: a *trained* policy grips deliberately and
  sustains far more contact than an untrained one, and the kernel's cost tracks
  live contacts, so the cost of tactile sensing rises as the policy improves.
- **The submodule is deliberately pinned** — the `leapXELA_model` copy vendored
  inside `leapXelaMjLab` sits at `4e8003f` (2026-06-11). Current `main` replaces `leapXela_generated_mjx_Box.xml` — same
  filename — with a **385-body, 1129-joint, 368-constraint flex hand**, whose 368
  taxels sit one per flex body. That breaks the splat's group-by-body assumption
  (the splat would degenerate to one taxel per contact) and would trigger the
  flex hypothesis H4 by accident, unmeasured. It would also invalidate the
  kernel's body-segment skip, which is where a factor of ~12 comes from. Bumping
  the submodule swaps the robot; do it deliberately or not at all.
- **One machine, one scene.** All numbers are DGX Spark GB10, cube-reorient. The
  crossover point is a property of this hardware and this scene, not a universal
  constant — and now that the kernel's runtime is data-dependent, so is the 188×.
  The contact regime is no longer a single point (§4.6 spans 0–25 contacts/env),
  but the scene is. The 4096-env *training* point is untested here; only 4096-env
  simulation is.
- **The stage split costs something to measure.** The physics/encoder breakdown
  comes from a second pass that synchronises between the two stages, draining the
  pipeline twice per step; it is therefore an upper bound on the fused cost. The
  headline throughput and cost-of-touch numbers come from the fused pass, with no
  inner synchronisation.
- **H5 is unresolved**, and the evidence runs against it: memory has failed to
  bind in every measurement taken (physics at 4096 envs, 1.37 GB; flex at 1024,
  1.22 GB). What was never re-measured is the *training loop*, which is what
  crashed the host in the first place — so the memory cap and subprocess
  isolation stay.
- **Native touch under motion is bounded, not resolved.** Its grasp-motion cost
  at N = 4096 is 1.08×, but the effect stays comparable to run-to-run variation —
  at N = 1024 the touch sweep reads nominally *faster* than physics-only. Flex is
  the one representation never priced under motion; its margin is such that
  nothing could close it.

## 7. Reproduction

### Environment

| | |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10, 48 SMs, sm_121, `aarch64`, ~121 GB **unified CPU/GPU** LPDDR5X |
| Kernel | `6.17.0-1029-nvidia` |
| Python | 3.12.3 |
| mujoco | 3.11.0 |
| mujoco_warp | 3.11.0 |
| warp-lang | 1.16.0 |
| torch | 2.13.0+cu130 |
| triton | 3.7.1 (installed with torch) |
| numpy | 2.5.2 |
| CUDA | 13.0 (nvcc 13.0.88) |
| matplotlib | 3.11.1 (figures only) |

> **Unified memory warning.** There is no separate VRAM;
> `nvidia-smi --query-gpu=memory.total` returns `[N/A]`. An oversized allocation
> does not raise `CUDA out of memory` — it exhausts host RAM, the OOM killer
> starts killing system daemons, and the machine wedges (this happened once, on
> 2026-08-13). **Cap every run.** Rendering is headless via EGL: set
> `MUJOCO_GL=egl` before importing mujoco (the scripts here do it themselves).

### Setup

```bash
git clone --recurse-submodules <this repo> && cd Mujoco
python3 -m venv .venv
.venv/bin/pip install mujoco==3.11.0 mujoco-warp==3.11.0 numpy==2.5.2 matplotlib==3.11.1
# torch must be the aarch64 CUDA 13.0 build (2.13.0+cu130 here), not the default
# wheel; Triton comes with it
```

The submodule URL is SSH (`git@github.com:...`); without a key on the box, use
`git config --global url."https://github.com/".insteadOf git@github.com:`.

### Correctness first

```bash
# batched encoder vs the banked CPU oracle (CPU only, seconds)
PYTHONPATH=third_party/leapXELA_model:explore \
  systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
  .venv/bin/python tests/test_encoder_oracle.py

# the Triton kernel vs the oracle, vs the torch encoder, edge cases, determinism
systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
  .venv/bin/python tests/test_triton_kernel.py

# either GPU tactile path vs the CPU reference, at a small env count
systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
  .venv/bin/python benchmarks/harness.py --num-envs 4 --tactile splat \
  --encoder triton --verify
```

`harness.py` puts `src/`, `explore/` and `third_party/leapXELA_model` on
`sys.path` itself, so it needs no `PYTHONPATH`; the test's `PYTHONPATH` above
matches its own docstring.

### The profiling

```bash
CAP="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0"
$CAP .venv/bin/python profiling/profile_encoder.py --mode bandwidth
$CAP .venv/bin/python profiling/profile_encoder.py --mode bench --num-envs 4096 \
     --bandwidth --iters 10 --profile-calls 3
$CAP .venv/bin/python profiling/profile_encoder.py --mode fusion --num-envs 4096 --iters 10
```

`--mode ncu` exists but `ncu` itself cannot read a counter on this box; see §4.8.

### The sweeps

`scale_sweep.py` runs `harness.py` once per env count in its own
`systemd-run --scope` with a hard `MemoryMax`, so an out-of-memory kills a child
rather than the host. A failure at some N is a result — the ceiling — not a crash.

```bash
# H1 control: bare physics, N = 1 .. 4096
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile none --tag physics_only_dt01

# H2: physics + the dense torch splat encoder, same ladder
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile splat --tag splat_dt01

# H6: the same ladder with the Triton kernel
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile splat --encoder triton --tag splat_triton_dt01

# the other two rigid representations
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile touch --touch-scene inject --tag touch_dt01
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile touchgrid --touchgrid-collide object \
  --nconmax-per-env 256 --njmax-per-env 2048 --tag touchgrid_object_dt01

# any of the above under real grasp motion: add --motion grasp, prefix the tag
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile splat --encoder triton \
  --motion grasp --tag grasp_splat_triton_dt01

# flex is a different body tree and gets its own probe, not a sweep
.venv/bin/python benchmarks/flex_probe.py --num-envs 1024 --njmax 4096

# the contact-budget law: N and njmax fixed, only the allocation varies.
# Two series, because the claim is that the geometry is irrelevant.
.venv/bin/python benchmarks/contact_budget_sweep.py \
  --tactile none --tag none_n1024
.venv/bin/python benchmarks/contact_budget_sweep.py \
  --tactile touchgrid --touchgrid-collide off --tag grid_off_n1024
```

Each writes `benchmarks/results/scale_sweep_<tag>.csv`. Run inside `tmux` so a
dropped SSH connection does not kill the job. **Run one GPU job at a time** — a
contended measurement in this project once read 4× slow and looked like a
regression. The `_dt01` suffix is a convention, not a flag: the harness fixes
`mjm.opt.timestep = 0.01` itself, and the suffix keeps these files distinct from
the superseded `dt = 0.001` sweeps banked under the bare tag.

`benchmarks/results/kernel_bench.csv` — the encoder-only eager /
`torch.compile` / Triton comparison in §4.5 — is banked from the kernel bring-up
run of 2026-08-13. Its driver script is not checked in, so unlike the three
sweeps above it cannot be regenerated by a command in this repository; the CSV is
the record. `tests/test_triton_kernel.py` reproduces the correctness and
determinism half of that run.

### The figures

```bash
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  .venv/bin/python analysis/plots.py
```

Regenerates all ten PNGs in `analysis/plots/` from the banked CSVs alone. No
number on any figure is hardcoded — crossover, saturation knee, cost of touch,
kernel speedups, the Amdahl ceiling and the curve through it are all recomputed at
plot time, so re-running the sweeps updates the titles too. The script prints the
derived headline numbers so a figure can be checked against the data without
opening an image. See `analysis/README.md`.

## 8. Repository layout

```
├── README.md                  # this file
├── report.tex                 # the write-up: four representations, kernel, threats
├── HYPOTHESES.md              # H1-H6 + scope decisions D1-D9, registered before measuring
├── PROJECT_LOG.md             # durable running record: what was done, decided, learned
├── third_party/
│   ├── leapXelaMjLab/         # submodule: the supervisor's GPU training stack
│   └── leapXELA_model/        # submodule: hand models + the CPU reference encoder
├── explore/                   # model archaeology: body map, taxel injection, oracle
│                              #   fixtures, grasp motions, splat-vs-flex
├── src/
│   ├── encoders/taxel_torch.py        # the dense baseline
│   ├── kernels/triton/taxel_triton.py # the two-pass gather kernel
│   └── mjlab_tactile/taxel_term.py    # the mjlab ObservationTermCfg (§4.10)
├── benchmarks/                # harness.py (4 representations, 2 motions),
│                              #   scale_sweep.py, contact_budget_sweep.py,
│                              #   flex_probe.py, mjlab_*.py, results/
├── contrib/mjlab-site-pos/    # the upstream mjlab fix, written up for a PR
├── profiling/                 # profile_encoder.py -> RESULTS.md, optables, nsys reports
├── tests/                     # test_encoder_oracle.py, test_triton_kernel.py
└── analysis/                  # plots.py -> plots/*.png
```

Planned and not yet present: `src/kernels/cuda/`, direct instrumentation of the
tactile stage's pre-processing (§4.8), and a contact-budget scan across several
env counts and a second scene — the affine law now has a banked sweep behind it,
but both series were taken at one N on one scene, so whether the slope is a
constant of the machine or a function of N or of the model's DoF is open.

## 9. Attribution

The hand models, the touch-sensor and touchgrid model variants, and the GPU
training stack ([`leapXelaMjLab`](https://github.com/mohammad200h/leapXelaMjLab))
are the supervisor's work
([`mohammad200h/leapXELA_model`](https://github.com/mohammad200h/leapXELA_model)),
as are the seven grasp patterns of `sparsh-skin-sim`, whose constants are used
verbatim in the motion driver and the widened oracle. Both repos are vendored
here as submodules (via forks under `AfthabShiraz/`) and are unmodified.

**The CPU reference encoder `VirtualTaxelSensor` (`leapxela/touch_sensor.py`) is
the work of Pratik Ingle (`pratik-ingle`)**, contributed in commit `5f305a8` on
2026-08-03 — earlier versions of this README and of `report.tex` attributed it to
the supervisor, which was wrong. It is the semantic oracle for everything
measured here, so the attribution matters. Much of the recent sensor-modelling
and dataset work in those repos is his.

This repository adds the model reconciliation and taxel injection, the benchmark
harness and the four representation paths, the batched encoder, the profiling
study, the Triton kernel, the mjlab observation term, and the measurements above.
