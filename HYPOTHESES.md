# Hypothesis register

**Registered 2026-08-13, before any throughput measurement was taken.**

The point of this file is that it is written first. Predictions made after seeing
the data are not predictions. Each hypothesis gets a resolution entry — confirmed,
refuted, or unresolved — and refuted ones stay in the document.

---

## Scope decisions locked at registration

These are choices, not findings. They are recorded so results can be read against
them, and so nothing here is silently revised later.

| # | Decision | Rationale |
|---|---|---|
| D1 | **Target model:** the rigid hand mjlab loads — `leapXelaMjLab` submodule pinned at `4e8003f`, 17 bodies, no flex. | It is what the supervisor's 4096-env training actually uses. |
| D2 | **The submodule stays pinned.** Current `main` replaces that file with a 385-body / 1129-joint / 368-constraint flex hand under the same filename. | Bumping it would silently swap the robot mid-project and trigger the flex hypothesis (H4) by accident. Flex is measured deliberately, as its own representation, or not at all. |
| D3 | **Taxel placement** uses `BODY_MAP` (`explore/taxel_map.py`), derived from the kinematic tree and validated by body-local mesh bounding box. 8 of 12 bodies match to 0.000 mm. | Established without external input; see `PROJECT_LOG.md` §1.5b. |
| D4 | **Known limitation, accepted:** the four fingertips are a geometrically different part between the two model lineages (~10.3 mm longer in the RL lineage), so the lateral placement of 136 of 368 taxels is unverified against the physical robot. | Affects sim-to-real fidelity. Does **not** affect throughput, bottleneck location, or kernel correctness — this project measures performance. To be resolved with the supervisor when convenient, not as a blocker. |
| D5 | **Correctness is defined against the CPU reference encoder** (`VirtualTaxelSensor`), not against the physical robot: given identical contacts, every implementation must reproduce it to an agreed tolerance. | Self-contained, and the strongest correctness claim available regardless. |
| D9 | **Benchmarks run at `sim_dt = 0.01`, matching `leapXelaMjLab/env_cfg.py`** (`ctrl_dt=0.05`, `decimation=5`). The harness sets `mjm.opt.timestep` explicitly. | The scene includes `leapXela_generated_mjx.xml`, which declares `timestep=0.001` — **10× smaller than the supervisor trains at**. Left uncorrected, our env-steps/s would not be comparable to his training throughput (10× the steps for the same simulated robot time). Found and corrected 2026-08-13. The earlier `dt=0.001` sweeps are retained as `*_dt01`-less CSVs; all three used identical settings so their *ratios* were always sound. |
| D6 | **Task, reward, RL algorithm inherited unmodified.** The observation space is the only thing that changes, plus whatever the profiler forces. | Preserves comparability with the supervisor's baseline. |
| D8 | **Implementation ladder is Python → Triton → CUDA.** The plan's C++/OpenMP stage (§5 Phase 4) is **dropped**. | Afthab's call, 2026-08-13. The C++ stage taught CPU parallel decomposition and false sharing; that is real but it is not on the critical path to the GPU result, and the Triton→CUDA pair already covers the kernel-authoring skill. Revisit only if a CPU baseline is needed for the write-up. |
| D7 | **Ceiling on env count is the DGX Spark's unified memory**, not a chosen number. The sweep runs to whatever N fits under a memory cap; 4096 is not assumed reachable. | See `PROJECT_LOG.md` §1.1. |

> **D7 REVISED, 2026-08-13, same day.** Measured: **4096 envs of bare physics use 0.92 GB and run at 129,682 env-steps/s.** Memory is nowhere near binding — the earlier assumption that 4096 was out of reach on the Spark was wrong *for physics*. The sweep completed all 13 rungs with zero failures.
>
> This does **not** retract §1.1: the host crash was real. But it was not caused by physics memory at 4096. Plausible causes, none yet confirmed: the full mjlab/RSL-RL training loop (policy, optimiser, rollout buffers, WandB) rather than the sim; or the same `njmax` scaling mistake made in the first harness (below), which inflates per-world constraint memory and compute by a factor of N.
>
> **Revised stance:** keep the memory cap and subprocess isolation — they cost nothing and they caught this safely — but do not scope the project around an assumed low ceiling. Re-measure the ceiling with the *training loop* attached before concluding anything about 4096-env training.

---

## Findings from widening validation coverage (2026-08-14)

`explore/05_grasp_motions.py` ports the supervisor's seven grasp patterns from
`sparsh-skin-sim/util/motion_util.py` (constants copied verbatim, attributed in
the module docstring) and regenerates the oracle as
`explore/out/oracle_fixture_v2.npz`: **632 frames, 56 trials**, versus 30 frames
before. Coverage:

| | v1 | v2 |
|---|---|---|
| taxels ever live | 37/368 | **176/368** |
| fingertips | 2/136 | **57/136** |
| medial links | 0/48 | **0/48** |
| contacts/frame max | 14 | 19 |
| shear range | ±2.5 N | **±5.4 N** |

Both implementations still validate. All 11 tests pass.

### F1 — The encoder's gate is discontinuous, and float32 lands on it

Widening coverage exposed a **4.6% relative error** invisible to v1. It is *not*
on the fingertips (those are clean at ~6e-6) — it is on the **palm**, and it is
**numerical, not indexing or placement**.

The reference gates each splat with hard thresholds (`dist_sq > cutoff²`,
`|local_z| > cutoff`) and then renormalises by the *surviving* weight sum. In
this scene the palm taxel plane sits at `|local_z|` = **10.0000 mm** against a
10 mm cutoff — measured margins of **0.0 to 0.2 µm**. float32 cannot resolve
that, so a taxel falls on the opposite side of the gate from the float64
reference; losing one taxel from a 2–5 taxel splat rescales the survivors. Hence
a clean, identical 4.592e-2 on all three channels rather than noise.

Control: the *same* torch code in float64 collapses the error to **4.16e-07**.
Both implementations are correct; the algorithm is discontinuous and this scene
sits on the discontinuity.

**Implication for the supervisor:** any float32 GPU port of `VirtualTaxelSensor`
will disagree with the float64 CPU reference by ~5% relative on palm taxels
whenever the palm is loaded. That is a property of the gate, not of the kernel.
Tests now assert a tight float64 bound (1e-6, the real semantic claim) plus a
documented loose float32 bound; Triton-vs-torch at matched precision stays at
9.5e-7.

### F2 — 48 medial taxels are structurally unreachable in this scene

Every cube contact on `if_md`/`mf_md` is **11.50 mm** out of plane — a constant,
above the 10 mm cutoff. 0 of 32 medial contacts reached a taxel. **No motion
fixes this**; the pads sit on a different face of the link from the one the cube
presses. Verified as *not* a lineage artefact: the taxel-to-surface geometry is
**identical in both model lineages** (palm 1.00 mm, proximal 1.62 mm, medial
2.04 mm), so this is a property of the supervisor's own layout, not of our
transplant.

### F3 — D4 quantified: the fingertip mismatch is a knife-edge

Fingertip contacts sit **9–10.8 mm** out of plane — straddling the gate — and
**33–49% produce nothing**. The nearest in-plane taxel is 6 mm away laterally.
This is the signature of the cube touching the RL tip's extra ~10.3 mm of distal
length, which carries no taxels. Fingertip readings are therefore hypersensitive:
a sub-millimetre change to tip geometry flips taxels between firing and silent.
`explore/06_visualise_gate.py` and `07_grasp_video.py` render this directly —
three of four fingertips register contacts and fire zero taxels.

**One deviation from the supervisor's constants, flagged in code:**
`thumb_grip_fraction` 0.35 → 0.70 for half the trials. His 0.35 is a fraction of
*ctrl range*, which on this hand leaves the thumb more open than the scene's own
`home` pose, so it never opposes the cube and all 62 thumb taxels read zero. 4 of
8 trials per pattern still run his default profile verbatim.

**Remaining dark taxels (192):** 48 structurally unreachable medial + 51 thumb +
79 palm/fingertip/proximal that this scene's contact patches never visit. Max
contacts is 19 against the 48-slot budget — a one-cube scene does not saturate it.

---

## H1 — Physics dominates when tactile is off

With no tactile observation, the physics step accounts for the majority of
per-step wall-clock at every reachable env count, and throughput scales close to
linearly in N until the GPU saturates.

*Reproduces the supervisor's existing regime. This is the control — if it fails,
the harness is wrong, not the system.*

**Resolution (2026-08-13): scaling half CONFIRMED; dominance half still open.**

`benchmarks/results/scale_sweep_physics_only.csv`, MuJoCo Warp, cube-reorient
scene, hand closed to 60% so ~4.3 contacts/env are live, zero contact overflow
at every N:

| N | env-steps/s | µs/env-step | ms/step | peak GB |
|---|---|---|---|---|
| 1 | 268 | 3731.5 | 3.73 | 0.82 |
| 16 | 4,051 | 246.8 | 3.95 | 0.82 |
| 128 | 32,164 | 31.1 | 3.98 | 0.82 |
| 256 | 40,289 | 24.8 | 6.35 | 0.82 |
| 1024 | 78,550 | 12.7 | 13.04 | 0.82 |
| 4096 | **129,682** | **7.7** | 31.58 | **0.92** |

Scaling is **perfectly linear to N=128** — `ms/step` is flat at ~3.95 ms while
throughput doubles at every rung, i.e. 128 environments cost the same wall-clock
as one. Past N=128 the GPU saturates and `ms/step` starts climbing, but
throughput still improves sublinearly out to 4096. Per-env cost falls 484× from
N=1 to N=4096.

**GPU saturation point: N ≈ 128–256** on this hardware, for this scene.

The *dominance* claim (physics is the majority of step time) is not yet tested —
that needs the per-stage breakdown, which needs an encoder to compare against.

⚠ **D7 is refuted for physics-only** — see the decision table above.

---

## H2 — A naive splat encoder overtakes physics somewhere in N ∈ [64, 1024]

Ported faithfully but without optimisation, the splat encoder's share of step
time grows with env count and crosses over the physics step within that range,
after which throughput is encoder-bound.

*The crossover point is the headline number. The range is a genuine guess.*

**Resolution (2026-08-13): CONFIRMED. Crossover measured at N = 256, inside the predicted [64, 1024].**

`benchmarks/results/scale_sweep_splat.csv` vs `..._physics_only.csv`, identical
scene and settings, ~4.3 live contacts/env, zero contact overflow at every N:

| N | physics-only/s | splat/s | slowdown | physics ms | encoder ms | encoder % |
|---|---|---|---|---|---|---|
| 1 | 268 | 228 | 1.17× | 3.75 | 0.60 | 13.8% |
| 16 | 4,051 | 3,293 | 1.23× | 4.15 | 0.75 | 15.2% |
| 64 | 16,055 | 11,259 | 1.43× | 4.09 | 1.62 | 28.4% |
| 128 | 32,164 | 11,881 | 2.71× | 6.99 | 3.82 | 35.3% |
| **256** | 40,289 | 17,723 | 2.27× | 6.99 | **7.54** | **51.9%** ← crossover |
| 512 | 62,364 | 22,005 | 2.83× | 8.82 | 14.53 | 62.2% |
| 1024 | 78,550 | 25,555 | 3.07× | 11.57 | 28.43 | 71.1% |
| 4096 | 129,682 | 28,177 | **4.60×** | 33.33 | **112.26** | **77.1%** |

**The bottleneck migrates, as predicted.** At small N the encoder is a steady
~14% tax. Past N≈64 its share climbs monotonically; at N=256 it overtakes
physics; by N=4096 it is 77% of step time and touch costs a **4.6× throughput
penalty**.

Mechanism: physics scales sublinearly — **8.9×** more time for **4096×** the work
(3.75 → 33.33 ms, N=1→4096) — while the naive encoder scales close to linearly
(0.60 → 112.26 ms, **187×**). It is a dense `B × C × T` formulation — every env × 48 contacts × 368
taxels — so its work grows in proportion to N with no reuse. Physics amortises;
the encoder does not.

**Amdahl ceiling, stated before optimising:** the encoder is 77.1% of step time
at N=4096, so eliminating it *entirely* caps end-to-end speedup at
**1 / (1 − 0.771) = 4.37×**. Any kernel result must be reported against that
bound.

Memory never bound: torch peak 4.06 GB at N=4096, process peak 1.72 GB steady.

---

## H3 — The geom-based touchgrid degrades throughput from inside the physics step

The binary touchgrid (361 extra collision geoms) costs throughput even when its
readout is excluded, because it inflates contact generation against a static
budget (`nconmax=48`, `njmax=120`). Expect either measurable slowdown inside the
physics step, silently dropped contacts, or both.

*Prediction added at registration: contacts will be dropped before throughput
degrades noticeably, making this a correctness problem that presents as a
performance one.*

**Resolution:** _unresolved_

---

## H4 — Flex tactile is viable on Warp but impractical at scale

The flex/bubble representation (385 bodies, 1129 joints, 368 equality constraints
per hand) runs on MuJoCo Warp but costs enough throughput to be impractical for
RL at N ≥ 1024.

*Stretch. "It does not fit in memory" is a legitimate result, and on unified
memory it is a likely one.*

**Resolution:** _unresolved_

---

## H5 — Unified memory moves the ceiling, not just the throughput *(added for this hardware)*

On the DGX Spark's shared CPU/GPU memory, the maximum feasible env count — not
the per-step time — is the binding constraint for tactile representations, and
the memory cost per env is dominated by the contact arrays rather than the
368×3 taxel buffer.

*The taxel buffer is 1104 floats/env ≈ 4.4 KB — small. Contact arrays at
nconmax=48 are the suspected driver. Untested.*

**Resolution:** _unresolved_

---

## H6 — The gather reformulation beats the scatter, and determinism is why it matters

Reformulating the encoder from scatter (loop contacts, atomically accumulate into
taxels) to gather (one block per (env, taxel), loop that env's contacts,
accumulate in registers) is faster *and* removes run-to-run non-determinism. The
determinism will turn out to matter more than the speed for validating against
the oracle.

**Resolution (2026-08-13): conclusion SUPPORTED, stated mechanism REFUTED,
determinism half UNRESOLVED.** See `profiling/RESULTS.md`.

The recommended design (two-pass gather, recomputing weights) survives — but not
for the reason H6 gives.

*Refuted:* **atomics never appear in the profile.** The encoder is not slow
because a scatter contends for taxel slots; contention was never the cost. It is
slow because it writes **19.99 GB per call** at N=4096, of which 99.3% is
multiplied by zero *after* being written to DRAM. Measured 3.92 live contacts/env
against a 48-slot budget, then the body mask keeps 1/12 of taxels: 72.4M
(env, contact, taxel) triples computed, ~492k carry signal.

*Confirmed:* bandwidth is the limiter, so trading FLOPs for traffic is right.
Arithmetic intensity **0.192 FLOP/byte** against a measured ridge point of
**70.7 FLOP/byte** — 368× to the left. Running at 183.9 GB/s (71% of the measured
260.8 GB/s ceiling) and 0.19% of compute peak. Launch overhead is 0.18% of the
call (58 launches × 3.34 µs), so **any design justified on "fewer kernel
launches" is justified on the wrong grounds.**

*Also refuted:* the one-pass scatter alternative, twice over. `weights /= sum`
reduces over the taxel axis, so scatter cannot be one-pass either — and having
paid the two-pass cost it buys no traffic reduction while adding
non-determinism.

*Determinism — now MEASURED (2026-08-13, `tests/test_triton_kernel.py`):*
**20/20 runs bit-identical** at B=1024 via `torch.equal` over 1,130,496 floats.
The gather design delivers what it promised. Previously recorded as unmeasured.

**BUILT AND MEASURED — `src/kernels/triton/taxel_triton.py`:**

| N | eager | torch.compile | **Triton** | vs eager | vs compile | GB/s | % of ceiling |
|---|---|---|---|---|---|---|---|
| 128 | 2.887 | 1.282 | **0.038** | 74.9× | 33.3× | 110.3 | 42.3% |
| 1024 | 27.340 | 11.196 | **0.132** | 207.5× | 85.0× | 258.1 | 99.0% |
| 4096 | 109.525 | 45.536 | **0.582** | 188.2× | 78.3× | 233.8 | 89.6% |

Traffic 19.99 GB → **136 MB** at N=4096, against the 103.8 MB compulsory floor
(1.31× the floor). Peak torch allocation 4.095 → 0.183 GB (22×). Correctness
gated timing: 1.94e-06 max abs vs the oracle, 1.34e-07 vs the real Warp contact
stream at N=64, no NaNs.

This **exceeds the 20–40× projection** — the discounts it assumed (register
spills, load imbalance, 48-SM tail) did not materialise. Cross-checked: wall-clock
over 100 calls 0.573 ms vs event-timed 0.582 ms.

⚠ **New caveat — runtime is now data-dependent.** The dense baseline's cost was
independent of contact count; the sparse kernel's is not. At N=4096: **0.54 ms at
1 live contact/env, 0.57 at the real 4.3, 0.73 at 12, 1.27 ms with all 48 slots
live.** The worst case is still 86× eager / 36× `torch.compile`, but the headline
figure is tied to the ~4.3 contacts/env this scene produces. A harder grasp would
cost more, and the profiling report's "synthetic == real to 1.0006" result no
longer transfers.

⚠ **Read the 96–99% GB/s figures at N=512–2048 with suspicion.** They are
modelled-bytes ÷ measured-time, not counter reads (`ncu` still blocked), so they
almost certainly reflect L2 reuse being charged as DRAM traffic rather than the
wall being approached. **89.6% at N=4096 is the defensible figure.**

**END-TO-END SWEEP, MEASURED (2026-08-13)** — `scale_sweep_splat_triton.csv`,
same scene and settings, kernel wired into the harness behind `--encoder triton`
(correctness re-gated in-harness at 1.49e-07):

| N | no touch | torch splat | **Triton splat** | cost of touch | vs torch |
|---|---|---|---|---|---|
| 64 | 16,055 | 11,259 | 13,939 | 1.15× | 1.24× |
| 256 | 40,289 | 17,723 | 33,301 | 1.21× | 1.88× |
| 1024 | 78,550 | 25,555 | 76,349 | **1.03×** | 2.99× |
| 4096 | 129,682 | 28,177 | **119,681** | **1.08×** | **4.25×** |

**REPLICATED AT THE SUPERVISOR'S TIMESTEP (2026-08-14).** All three sweeps re-run
at `sim_dt = 0.01` per D9 (`*_dt01.csv`). Contacts rise from ~4.3 to **~7.5 per
env** — larger steps mean deeper interpenetration — so this is a *harder* case
for the encoder, and the finding survives it:

| N | no touch | torch splat | Triton splat | cost torch | cost Triton |
|---|---|---|---|---|---|
| 64 | 15,999 | 11,264 | 13,960 | 1.42× | 1.15× |
| 256 | 38,310 | 18,486 | 34,715 | 2.07× | **1.10×** |
| 1024 | 78,349 | 25,148 | 77,361 | 3.12× | **1.01×** |
| 4096 | 128,061 | 28,006 | **116,961** | **4.57×** | **1.09×** |

**Incidental finding — `njmax=120` is marginally tight at `sim_dt=0.01`.** At the
supervisor's own timestep, **2 of 4096 worlds (0.049%)** raise `OverflowType.NEFC`;
Warp reports *"nefc overflow — please increase njmax to 124"* (observed up to
128). Contacts are nowhere near their limit: 7.1/env against the 48 cap, so this
is **constraint-row overflow, not contact overflow**. It does not appear at
`dt=0.001`. Minor at 0.05% of worlds, but it is the supervisor's configured value
and worth reporting to him. ⚠ The harness's `overflow_worlds` column counts *any*
overflow bit, so it must not be read as a contact-budget figure — and the plot
subtitle labelling it "contact overflow" is wrong and needs correcting.

**Cost of touch 4.57× → 1.09×, kernel gain 4.18×** — within 2% of the
`dt=0.001` figures despite 75% more contacts. The result is not an artifact of a
gentle contact regime. **Quote these numbers, not the `dt=0.001` ones**, since
only these are comparable to the supervisor's training configuration.

The original `dt=0.001` measurement, retained for reference:

**The cost of tactile sensing at 4096 envs falls from 4.60× to 1.08×** — from a
360% penalty to 8%. End-to-end gain 4.25× against the 4.37× Amdahl cap, i.e.
**97% of everything that was available**. The encoder's share of step time drops
from 77.1% to 6.5%.

⚠ **The bottleneck has migrated again — inside the tactile stage.** The harness
reports 2.393 ms for the encoder stage at N=4096, but the kernel alone measures
0.582 ms. The remaining ~1.8 ms is in the stage's *pre-processing*: fetching
contact forces from Warp (`mjw.contact_force`) and scattering the flat contact
array into the padded per-env `(B, C, ·)` layout. **Pre-processing is now ~75% of
the tactile stage.** This is the project's own thesis recurring one level down,
and it is the natural next profiling target — ahead of physics, and much cheaper
to attack. Not yet separately instrumented; the split above is inferred from the
two measurements and should be measured directly before being quoted.

⚠ **Amdahl is now the entire story.** At 188×, end-to-end reaches ~4.35× against
the 4.37× cap — versus 3.93× for a hypothetical 30× kernel. **The remaining 10%
of end-to-end throughput lives entirely in the physics step.** Further tuning of
this kernel is not worth the days; the honest next profiling target is physics.

*Honest caveat on recompute-vs-store:* in a fully body-sparse layout the stored
weight buffer is only ~2 MB, so store's bandwidth penalty largely evaporates.
Recompute then wins on simplicity and footprint rather than on traffic.

**Decisive experiment:** `torch.compile` on the unmodified encoder changes zero
FLOPs, zero algorithm and zero occupancy strategy — it removes only DRAM round
trips — and yields **2.41× at N=4096** (independently re-run), numerically
identical at 4.8e-07. Traffic is the limiter.

**Actionable side finding:** that 2.41× is available today from a one-line
change, worth ~1.82× end-to-end.

**Projected kernel speedup: 20–40×** (traffic floor 103.8 MB → 0.43 ms;
realistic landing 1–3 ms).

⚠ **Amdahl reality check, decide before tuning:** cap is 4.37×. A 20× kernel
gives 3.74×, 30× gives 3.93×, 100× gives 4.23×. **Going 30× → 100× buys 7.6%
more end-to-end throughput for 3.3× more kernel work.** The headline saturates
at 20–30×; past that the honest next target is physics, not the encoder.

⚠ **Tooling limitation:** `ncu` returns `ERR_NVGPUCTRPERM` on every section —
the driver sets `NVreg_RestrictProfilingToAdminUsers=1` and there is no
passwordless root. **There is no Nsight-reported limiter and no Nsight-measured
DRAM throughput in this analysis.** Substituted: an empirically measured
saturating-copy ceiling (260.8 GB/s read, 240.7 copy, 237.6 triad), an SGEMM
compute ceiling (18.44 TFLOP/s, TF32 off), and `modelled bytes ÷ measured kernel
time` for every achieved-GB/s figure. The fusion experiment exists precisely
because it tests the same conclusion without counters. `nsys` works and confirms
the kernel-level picture.
