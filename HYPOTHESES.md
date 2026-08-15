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
| D4 | **Known limitation, accepted:** the four fingertips are a geometrically different part between the two model lineages (~10.3 mm longer in the RL lineage), so the lateral placement of 136 of 368 taxels is unverified against the physical robot. | Affects sim-to-real fidelity. Does **not** affect throughput, bottleneck location, or kernel correctness — this project measures performance. To be resolved with the supervisor when convenient, not as a blocker. **⚠ Twice revised since: F3 shows a functional consequence (fingertip contacts fire no taxels), and F4 shows the cause is one revised CAD part rather than two divergent models — so this has a definite answer, not an open-ended reconciliation.** |
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

### F4 — D4 is one revised part, not two divergent models (2026-08-15)

Established by hashing the mesh assets rather than comparing compiled models.
**17 `.stl` filenames are shared between the lineages; all 17 have identical
triangle counts and 10 are byte-for-byte identical.** Independent CAD exports do
not produce identical files — these are two exports of one project.

The fingertip is the **only** component sharing no filename: RL
`fingertop_unfied.stl` is 72.31 mm on its long axis, tactile `fingertip_fix.stl`
is 61.98 mm. The 10.33 mm difference reproduces the compiled-model measurement
exactly, now traced to source geometry.

**This changes D4's status from "accepted limitation" to "open question with a
definite answer".** D4 was registered on the assumption that two lineages had
simply diverged and reconciling them was open-ended. In fact one part was revised,
so *someone knows which revision is the physical robot*. Full detail and the exact
form to ask Hamid: `PROJECT_LOG.md` §1.5b-bis.

⚠ Also corrects §1.5b's stated reason for validating by bounding box rather than
centroid. The claim was that the RL meshes are decimated; they are not (identical
triangle counts). Corresponding meshes are re-exports whose vertices differ by
≤1.5 mm — which still moves a centroid, so the method was right and the reason
recorded for it was wrong.

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

**Resolution (2026-08-14): CONFIRMED in its conclusion, REFUTED in its mechanism.
The touchgrid does cost throughput from inside the physics step — 4.3× at
N=4096 — but not for the reason predicted. The contacts it generates are free.
What costs is the contact budget it forces you to *allocate*.**

First, two corrections to the registered wording:

* **300 patch geoms, not 361.** 361 is the count of `<geom ` lines in
  `robot_touch_sensor_array_binary_touchgrid_generated_mjx.xml`, which includes
  visuals, the coarse collision boxes and class defaults. The sensor grid itself
  is 300 geoms, measured off the compiled model. Injected into
  `scene_mjx_cube.xml` this takes the scene from 71 geoms to 371.
* **"Silently dropped contacts" understates it.** At the supervisor's
  `nconmax=48 / njmax=120`, mujoco_warp does not drop contacts and carry on. It
  runs past its buffers and dies: `Warp CUDA error 700: illegal memory access` in
  `ccd_kernel` (`collision_convex.py:1351`) and `_qfrc_constraint_from_grad`.
  The touchgrid cannot be run at the supervisor's budget at all.

Measured requirement, CPU reference, closing grasp: **ncon 232 / nefc 944**,
against the bare scene's 24 / 112. A 5× contact and 8× constraint shortfall.

**The contact explosion is intrinsic to the representation.** At the 232-contact
peak the breakdown is **223 patch-cube, 0 patch-patch, 0 patch-hand, 0
patch-floor**. 300 small boxes tiling the hand against a cube generate ~223
simultaneous contacts where the hand's 37 coarse collision boxes generate ~9.
Masking the patches to collide only with the cube (`--touchgrid-collide object`,
`contype=2 conaffinity=0`) changes nothing, because they were never touching
anything else.

> *A discarded explanation, recorded because it was wrong in an instructive way.*
> The first reading blamed D4 — patches injected into the RL lineage's
> differently-shaped links, interpenetrating them — on the evidence that the
> donor model peaks at only 38 contacts. **That control was invalid: the donor
> model standing alone has no cube in it**, so it was measuring self-collision
> with nothing to grip. Classifying the contacts by what they touch is what
> settled it.

### The decomposition — and where the cost actually is

Four sweeps, N=4096, all on `scene_mjx_cube.xml`, all at `sim_dt=0.01`:

| variant | geoms | nconmax/njmax | env-steps/s | vs bare |
|---|---|---|---|---|
| physics only | 71 | 48 / 120 | 128,061 | — |
| touchgrid, `collide=off` | 371 | 48 / 120 | 124,963 | **−2.4%** |
| touchgrid, `collide=off` | 371 | 256 / 2048 | 28,926 | **−77.4%** |
| touchgrid, `collide=object` | 371 | 256 / 2048 | 29,440 | **−77.0%** |

Read down that table: 300 extra collision geoms cost **2.4%**. Turning their
contacts on, with the budget held fixed, costs **nothing** — `object` is if
anything a hair faster than `off`, well inside run-to-run variation. The entire
4.3× collapse sits between rows 2 and 3, which differ only in how much contact
space was allocated.

### Which budget, and how it scales

N=1024, `collide=off` so the live contact count is pinned at 7.4/env in every
row — only the allocation changes:

| nconmax | njmax | env-steps/s |
|---|---|---|
| 48 | 120 | 82,422 |
| 48 | 2048 | 100,258 |
| 256 | 120 | 24,012 |
| 256 | 2048 | 26,670 |

**`njmax` is not the problem; `nconmax` is.** Raising njmax 120→2048 costs
nothing (the 100,258 row is *faster*, within noise). Raising nconmax 48→256 costs
71% of throughput with an identical number of real contacts.

Scan of `nconmax` alone (N=1024, njmax=120, 7.4 contacts/env throughout):

| nconmax | 48 | 96 | 128 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|
| ms/step | 11.71 | 19.45 | 25.43 | 34.93 | 40.08 | 61.23 |

Linear: **ms/step ≈ 5.3 + 0.145 × nconmax**, R² by eye ≈ 1. The step cost is an
affine function of the *allocated* contact budget and is blind to occupancy.

**BANKED AND REPRODUCED, 2026-08-15.** The scan above was a handful of one-off
runs whose numbers lived only in this file. `benchmarks/contact_budget_sweep.py`
now runs it properly — one memory-capped child per budget, a CSV, a least-squares
fit — and it is run **twice**, on two scenes differing fivefold in geom count, to
test whether the cost really is the allocation and nothing else:

| nconmax/env | 48 | 96 | 128 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|
| bare scene, ms/step | 13.12 | 18.73 | 23.13 | 32.32 | 43.34 | 60.01 |
| grid `collide=off`, ms/step | 11.60 | 19.21 | 23.65 | 32.03 | 39.34 | 61.11 |

```
bare scene (71 geoms)         ms/step = 5.51 + 0.1428 x nconmax   R² = 0.9978
grid, collide off (371 geoms) ms/step = 4.73 + 0.1436 x nconmax   R² = 0.9955
```

**The two slopes agree to 0.6%** across a 5× difference in geometry, and live
occupancy is pinned at **7.4–7.7 contacts/env at every one of the twelve points**.
That is the cleanest available statement of the mechanism: the cost attaches to
the allocation, not to the geometry and not to the occupancy. Throughput falls
78,045 → 17,062 env-steps/s over the scan (**4.6×**) with identical physics.

The original slope of 0.145 reproduces to within 1%. The intercept moves more
(5.3 → 5.51 / 4.73), which is expected — it is the noise-sensitive parameter of
the two and it is not the claim.

Plotted as `analysis/plots/J_contact_budget.png`, derived from the CSVs at plot
time like every other figure. **Still open:** both series are at *one* env count
on *one* scene, so whether the slope is a constant of the machine, a function of
N, or a function of the model's DoF is unestablished.

**Mechanism, from the source.** mujoco_warp sizes its launches off `d.naconmax`,
the allocation, not off the live count — `collision_convex.py:1250`,
`collision_primitive.py:1490`, `constraint.py:5327,5377`, `sensor.py:2514`,
`sleep.py:745`, and most expensively `smooth.py:2941`
(`dim=(m.nacttrnbody, d.naconmax, m.nv)`). Every step pays for every slot.

### What this means for the supervisor

`nconmax` is a **first-class throughput parameter**, not a safety margin. At
N=1024 the difference between `nconmax=48` and `nconmax=256` is 3.4× end-to-end,
with the same physics. Two consequences:

1. Any tactile representation that forces the contact budget up pays
   proportionally, *before* it computes a single taxel value. That is the real
   cost of the geom-based route, and it is structural.
2. `leapXelaMjLab`'s `nconmax=48` is worth confirming as deliberate and tight
   (open question 4). If it is padded, trimming it is free throughput; if it is
   too tight for some grasp, the failure mode on GPU is a crash, not a warning.

### Under grasp motion (2026-08-14) — the budget question sharpens

Re-run with `--motion grasp`, N=4096: the cube-only grid measures **27,942
env-steps/s** against a moving physics-only baseline of 126,637, a cost of
**4.53×** (frozen: 4.35×). Peak per-world contact count rises to **390**, against
the 232 measured at the frozen closing grasp, with no overflow at the 256-slot
allocation. The mechanism holds under motion: 151,669 patch-cube contacts and
**zero** patch-patch.

**Consequence: the budget a geom-based grid requires is a property of the
motion, not of the pose you happen to benchmark.** Sizing `nconmax` from a static
grasp under-provisions it by ~1.7× here — and the failure mode of
under-provisioning is the `ccd_kernel` illegal memory access above, not a warning.

**H3's registered prediction — "a correctness problem that presents as a
performance one" — was right in spirit and for the wrong reason.** It is a
correctness problem (it crashes) *and* a performance one (4.3×), but the two are
not the same effect: the crash comes from real contacts overrunning the budget,
and the slowdown comes from the allocation you make to stop the crash.

---

## H4 — Flex tactile is viable on Warp but impractical at scale

The flex/bubble representation (385 bodies, 1129 joints, 368 equality constraints
per hand) runs on MuJoCo Warp but costs enough throughput to be impractical for
RL at N ≥ 1024.

*Stretch. "It does not fit in memory" is a legitimate result, and on unified
memory it is a likely one.*

**Resolution (2026-08-14): CONFIRMED on both halves — flex runs on Warp, and it
is impractical. But the predicted failure mode was wrong: it is not memory, it
is time.**

Measured with `benchmarks/flex_probe.py` on
`leapXELA_model/scene_flex_sensor_Box.xml` — 387 bodies, 1120 joints, 18 flexes,
386 equality constraints, `sim_dt=0.01`, hand closed to the same
`CLOSE_FRACTION=0.6` grasp the rigid harness uses.

**Viability: yes.** `mjw.put_model` accepts the flex model without complaint.
This is the concrete confirmation of the plan's §2(c) argument — flex is
GPU-viable on Warp and definitively impossible on MJX-JAX
(`NotImplementedError('Flex not implemented for JAX backend.')`).

`put_data` requires `njmax ≥ 3030` per world and **says so cleanly, as a
`ValueError` at allocation time**. That contrast is worth keeping: the same class
of budget problem announces itself politely in flex and fatally in the touchgrid
(H3), which suggests the touchgrid crash is a mujoco_warp robustness gap rather
than an inherent property of overrunning a budget.

**Cost (gripping, `njmax=4096`, ~2.0 contacts/env):**

> **⚠ CORRECTED 2026-08-14, after the first version of this table was published
> internally.** The original figures were taken with `flex_probe.py` forcing
> `timestep = 0.01` to match `env_cfg.py`. **The flex model declares
> `timestep = 0.002` with 50 solver iterations, and does not survive 0.01**: at
> that step size MuJoCo prints `Nan, Inf or huge value in QACC ... the simulation
> is unstable` and the Kelvin-Voigt readout returns forces of order 1e5 N. Those
> numbers were measured on a misconfigured, intermittently diverging simulation
> and are superseded by the table below. A second, smaller error rode along with
> it: the settle was specified as a fixed 60 steps, which is 0.6 s at dt=0.01 but
> only 0.12 s at dt=0.002 -- not long enough for the hand to close, so the first
> corrected run reported zero contacts. Settle is now specified in simulated
> seconds.
>
> The conclusion is unchanged and in fact strengthened; only the magnitude moves.

Measured at the model's own `timestep=0.002` / `iterations=50`, gripping,
`njmax=4096`. Because a 0.002 s step advances one fifth as far as the rigid
sweeps' 0.01 s step, raw steps/s overstates flex 5x against them; the comparable
column rescales by `sim_dt / 0.01`.

| N | ms/step | raw steps/s | **equiv env-steps/s** | bare rigid at same N | ratio |
|---|---|---|---|---|---|
| 1 | 137.1 | 7 | **1** | 308 | ~220× |
| 16 | 195.1 | 82 | **16** | 3,897 | ~238× |
| 64 | 256.9 | 249 | **50** | 15,999 | **~320×** |
| 256 | 719.2 | 356 | **71** | 38,310 | **~540×** |

Solver iterations are not the cost: 50 vs 5 iterations at dt=0.002 measured
159.8 vs 159.1 ms at N=64 (contact-free). The timestep is.

A single flex world costs **128 ms/step against the rigid hand's 3.2 ms** — 40×
before any parallelism at all.

**It saturates early.** Throughput gains flatten well before the rigid hand's
do — 50 to 71 equivalent env-steps/s between N=64 and N=256 while per-step
latency nearly triples — and a separate measurement at the (unstable) larger
timestep showed it flat from N≈128 with per-step time then scaling exactly
linearly, i.e. a fully saturated GPU rather than a launch-bound one. The rigid
hand, by contrast, is still gaining throughput from N=1024 to N=4096.

**Where the time goes** (N=64, measured by ablation): ~37% the constraint solver
over 386 equality constraints, ~63% plain forward dynamics over 387 bodies and
1120 DoF, and **~1% contacts**. Touch has essentially nothing to do with it. A
kernel cannot help: all of it is inside `mjw.step`, none of it is code this
project controls, and the Kelvin-Voigt readout it does control is a few thousand
flops against ~200 ms.

**The predicted binding constraint was wrong.** H4 and the surrounding hardware
notes (§1.1, D7) expected flex to fail by exhausting the Spark's unified memory.
Peak RSS is **0.75 GB at N=1 and only 1.22 GB at N=1024** — the memory the plan
worried about never became relevant. The model is simply expensive to integrate:
1120 DoF and 386 equality constraints per hand, solved every step.

This is the second time D7's memory-ceiling assumption has been measured and
found not to bind (the first was bare physics at 4096 envs, 0.92 GB). The Spark's
unified memory is a real hazard for the *training loop* — the §1.1 host crash
happened — but it has not once been the limit for the simulation itself.

**Caveat on comparability, stated because it limits the claim.** The flex hand
lives in the tactile lineage and is a different body tree from the pinned rigid
RL model; there is no injection that turns one into the other, the way taxel
sites and patch geoms could be injected. So `ms_per_step` here is a fair
same-engine/same-dt/same-GPU order-of-magnitude comparison, but
`env_steps_per_sec` is **not** a like-for-like fourth bar for the
cost-of-tactile-sensing figure and must not be plotted as one.

**Verdict against the registered claim ("impractical for RL at N ≥ 1024"):**
confirmed, and by a wider margin than first reported. At N=256 flex delivers 71
equivalent env-steps/s against the rigid hand's 38,310 — **0.19%**, a factor of
~540 — and it has already stopped scaling. Reaching training scale is not a
matter of finding more memory or more patience.

*Not attempted: whether the flex skin's taxel readout adds anything on top of
this. There would be no point — the physics alone is ~540× over budget, so the
readout cannot change the conclusion. If flex ever becomes viable, that
measurement gets made then.*

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

**RE-MEASURED UNDER REAL GRASP MOTION (2026-08-14) — the frozen-pose caveat above
is retired, and the result improves.** Every sweep above was taken at one held
pose: ~7.5 contacts/env, constant, *identical in every world*. That flatters a
kernel whose cost tracks live contacts against a dense baseline that is blind to
them. `harness.py --motion grasp` (new `GraspDriver`) replaces it with the seven
patterns from `explore/05_grasp_motions.py`, world `w` on pattern `w % 7` at a
phase offset, so the batch spans the whole modulation. At N=4096 the per-world
contact count runs 0 to 25 with a population mean of 7.33/env.

| path | frozen | grasp motion | frozen cost | grasp cost |
|---|---|---|---|---|
| physics only | 128,061 | 126,637 | 1.00× | 1.00× |
| eager splat | 28,006 | 27,642 | 4.57× | 4.58× |
| Triton splat | 116,961 | **125,535** | 1.09× | **1.01×** |

Encoder stage 2.394 ms under motion vs 2.432 ms frozen; the eager baseline is
unchanged at 112.99 vs 112.19 ms, exactly as its contact-blind `B × C × T` shape
predicts.

**The Triton path improves, and the mechanism is the decomposition, not luck.**
Pass 2 is indexed by (env, taxel tile): a program walks its own env's contact
slots, finds them invalid and exits the inner loop. An idle world costs a scan of
its slot array rather than an evaluation, and with no atomics and no cross-world
synchronisation it does not stall a loaded world either. Under motion many worlds
sit below the frozen pose's uniform occupancy, so total work falls.

⚠ **1.01 is "not resolvable", not a bound.** At N=1024 the Triton path measures
76,891 against a physics-only 72,804 — nominally faster than doing no tactile
work at all. Under motion at large N the end-to-end cost is not distinguishable
from run-to-run variation. **The upper end remains unmeasured**: no condition
holds every world near the 48-slot cap, and the synthetic fully-occupied case
costs 1.27 ms, 2.2× the measured regime. A grasp that saturates the budget would
cost more than 1.01.

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
