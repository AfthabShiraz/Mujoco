# Tactile observations at training scale

**What does touch cost when you simulate thousands of hands at once?**

This repository measures the cost of hardware-faithful tactile sensing inside a
GPU-parallel MuJoCo Warp simulation of the LeapXELA hand, and locates the env
count at which the tactile encoder — not the physics — becomes the bottleneck.

**Headline result:** on an NVIDIA DGX Spark (GB10), a faithful batched port of the
368-taxel Gaussian-splat encoder costs a flat ~14% of step time up to N ≈ 16,
overtakes the physics step at **N = 256**, and reaches **77.1% of step time at
N = 4096**, where turning touch on costs a **4.60× throughput penalty**. Because
the encoder is 77.1% of the step, removing it entirely would gain at most
**4.37×** end to end — that is the Amdahl bound any future kernel work is
measured against, and it was stated before any optimisation was attempted.

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

Measurement infrastructure and the correctness-validated GPU baseline exist; the
optimised kernel does not yet. H1 and H2 are resolved (`HYPOTHESES.md`); H3–H6
are open. Everything below is measured on this hardware, for this scene, and is
reported as such.

## 3. What was built

| Component | File | What it is |
|---|---|---|
| **Tactile→RL body map** | `explore/taxel_map.py` | The two model lineages share **zero** body names. `BODY_MAP` is derived from the kinematic tree, disambiguated by the layout's own hardware patch names, and validated by body-local mesh bounding box: **8 of 12 bodies match to 0.000 mm**. Chain order is reversed from what the names suggest — `finger` is the *ring* finger — so mapping by name order silently swaps index and ring. |
| **Taxel injection** | `explore/03_inject_taxels.py`, `benchmarks/harness.py:build_scene_model` | 368 sites added to the pinned rigid mjlab model through **`MjSpec`** (67 → 435 sites). mjlab builds its hand through `MjSpec` too, so this transfers into an mjlab `spec_fn` nearly verbatim. Burial was measured by per-body ray cast: max 5.00 mm against the encoder's 10 mm patch-locality gate, so every taxel can still fire. |
| **Oracle fixture** | `explore/04_run_encoder.py` → `explore/out/oracle_fixture.npz` | 30 frames through a scripted grasp, storing the reference encoder's **inputs** (contact pos/frame/geoms/6-vector force, site poses, qpos) beside its 368×3 outputs. Any later implementation is validated without re-running MuJoCo, which sidesteps the Warp-vs-CPU contact discrepancy for pure encoder validation. Includes 8 `ncon = 0` frames. |
| **Benchmark harness** | `benchmarks/harness.py`, `benchmarks/scale_sweep.py` | One env count per memory-capped child process, because this box has unified CPU/GPU memory and an oversized run wedges the host rather than raising `CUDA out of memory`. Warmup, `wp.synchronize()` on both sides of the clock, contact counts and overflow reported every run. Two timing passes: fused (headline throughput) and stage-synchronised (the physics/encoder split). |
| **Batched torch encoder** | `src/encoders/taxel_torch.py` | Faithful batched port of `VirtualTaxelSensor.update`: dense `(B, C, T)` tensors, no fused kernels, no sparsity. Deliberately unoptimised — its cost is the baseline every later implementation is measured against. |
| **GPU contact plumbing** | `benchmarks/harness.py:SplatEncoder` | Warp keeps contacts in one flat `naconmax` array tagged by `worldid`, unordered; the encoder wants a dense per-env batch. Bucketing is done with a **stable sort**, not atomic counters — slot order then fixes the summation order, so the output is reproducible rather than wobbling at the 1e-7 level (relevant to H6). |

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
its control range and settled 250 steps so contacts are live (~4.3 contacts/env),
`nconmax = 48/env`, `njmax = 120/env`, 200 timed steps after 30 warmup steps.
**Zero contact-budget overflow and zero dropped contacts at every N.**

### H1 — physics dominates when tactile is off → scaling half **confirmed**

`benchmarks/results/scale_sweep_physics_only.csv`:

| N | env-steps/s | µs/env-step | ms/step | peak GB |
|---|---|---|---|---|
| 1 | 268 | 3731.5 | 3.73 | 0.82 |
| 16 | 4,051 | 246.8 | 3.95 | 0.82 |
| 128 | 32,164 | 31.1 | 3.98 | 0.82 |
| 256 | 40,289 | 24.8 | 6.35 | 0.82 |
| 1024 | 78,550 | 12.7 | 13.04 | 0.82 |
| 4096 | **129,682** | **7.7** | 31.58 | **0.92** |

Scaling is linear to N = 128 — `ms/step` is flat at ~3.95 ms while throughput
doubles at every rung, i.e. 128 environments cost the same wall-clock as one.
Past that the GPU saturates and `ms/step` climbs, though throughput still improves
sublinearly to 4096. Per-env cost falls 484× from N = 1 to N = 4096.

This also **refuted a scope decision**: D7 assumed the Spark's unified memory
would set the env ceiling. 4096 physics environments use 0.92 GB peak RSS. Memory
was nowhere near binding, and all 13 rungs completed with no failures. The host
crash that motivated D7 was real but was not caused by physics memory at 4096;
its cause is still unidentified, so the memory cap and subprocess isolation stay.

![Throughput vs env count](analysis/plots/B_throughput_scaling.png)

### H2 — a naive splat encoder overtakes physics in N ∈ [64, 1024] → **confirmed at N = 256**

`benchmarks/results/scale_sweep_splat.csv` against the physics-only sweep,
identical scene and settings:

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

![The bottleneck migrates](analysis/plots/A_bottleneck_migration.png)

The mechanism is straightforward. Physics amortises with scale — between N = 1
and N = 4096 it does 4096× the work for **8.9×** the time (3.75 → 33.33 ms) —
while the dense `B × C × T` encoder does not: every environment × 48 contact
slots × 368 taxels, work proportional to N with no reuse, 0.60 → 112.26 ms, a
factor of **187**.

![Encoder share of step time](analysis/plots/C_encoder_share.png)

![Slowdown factor](analysis/plots/D_slowdown_factor.png)

### The Amdahl ceiling — 4.37×

At N = 4096 the encoder is 77.1% of step time. Eliminating it **entirely** —
making tactile encoding free — therefore caps end-to-end speedup at

> 1 / (1 − 0.771) = **4.37×**

This is stated here, before any kernel exists, because it is the number any
future Triton or CUDA result must be reported against. A kernel that makes the
encoder 10× faster does not make the step 10× faster: it leaves 22.9% + 7.7% of
the original step, i.e. **3.3×** end to end. Worth knowing before spending weeks
on the kernel.

Memory never bound the tactile path either: torch peak 4.06 GB and process peak
1.72 GB at N = 4096.

## 5. Correctness

Timing is only worth reporting if the answer is right, and the documented failure
mode here is a *fast wrong answer* (all-zero taxels, or forces attributed to the
wrong finger). Four checks, all run before any timing is reported:

| Check | What it compares | Result |
|---|---|---|
| Batched torch encoder vs CPU reference | `tests/test_encoder_oracle.py`, all 30 oracle frames | **2.4e-06 max abs error** (5.5e-06 max relative), on peak per-channel forces of **5.4 N** |
| Batching vs per-frame | 30 frames padded into one B=30 batch vs one at a time | **bit-identical** (0.0e+00) — the padding and masking are inert |
| GPU path vs CPU encoder, identical contacts | `benchmarks/harness.py --verify`, check A (gate) | **1.5e-08 max abs error** |
| GPU path vs `VirtualTaxelSensor` on a CPU re-solve | check B (reported, not gated) | **3.8e-03** — Warp and CPU MuJoCo generate different contact sets for the same state; an engine difference, not an encoder bug |

The reference accumulates in float64 and casts to float32 on return; the batched
version runs in float32 throughout, so agreement is to float32 rounding rather
than bit-exact, and 2.4e-06 is consistent with that. `ncon = 0` frames produce
exact zeros, not NaN from a 0/0 normalisation — that is asserted, not assumed.

**Limitation, stated plainly:** the oracle fixture is narrow. Across its 30
frames only **37 of 368 taxels ever fire**, and **31 of those are on the palm** —
4 on proximal links and **2 on fingertips**. Contacts do occur on the finger
links, but each 4×4 pad covers one face, so a contact on an uncovered face
correctly yields zero weight and is skipped. The fixture therefore exercises the
palm path well and the fingertip path barely. Before any kernel is trusted
against it, the fixture needs a grasp that loads the fingertips.

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
  optimiser, no rollout buffer, no logging. Encoder share and the 4.37× ceiling
  are shares of the *simulation* step, and adding the RL machinery will lower
  them. The tactile term is also not yet wired into mjlab as an
  `ObservationTermCfg`.
- **The submodule is deliberately pinned** — the `leapXELA_model` copy vendored
  inside `leapXelaMjLab` sits at `4e8003f` (2026-06-11). Current `main` replaces `leapXela_generated_mjx_Box.xml` — same
  filename — with a **385-body, 1129-joint, 368-constraint flex hand**, whose 368
  taxels sit one per flex body. That breaks the splat's group-by-body assumption
  (the splat would degenerate to one taxel per contact) and would trigger the
  flex hypothesis H4 by accident, unmeasured. Bumping the submodule swaps the
  robot; do it deliberately or not at all.
- **One machine, one scene, one contact regime.** All numbers are DGX Spark GB10,
  cube-reorient, ~4.3 contacts/env. The crossover point is a property of this
  hardware and this scene, not a universal constant. The 4096-env *training*
  point is untested here — only 4096-env simulation is.
- **The stage split costs something to measure.** The physics/encoder breakdown
  comes from a second pass that synchronises between the two stages, draining the
  pipeline twice per step; it is therefore an upper bound on the fused cost. The
  headline throughput and slowdown numbers come from the fused pass, with no
  inner synchronisation.
- **H3–H6 are unresolved.** No touchgrid measurement, no flex measurement, no
  memory-ceiling result with the training loop attached, and no gather-vs-scatter
  kernel comparison.

## 7. Reproduction

### Environment

| | |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10, `aarch64`, ~121 GB **unified CPU/GPU** LPDDR5X |
| Kernel | `6.17.0-1029-nvidia` |
| Python | 3.12.3 |
| mujoco | 3.11.0 |
| mujoco_warp | 3.11.0 |
| warp-lang | 1.16.0 |
| torch | 2.13.0+cu130 |
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
# torch must be the aarch64 CUDA 13.0 build (2.13.0+cu130 here), not the default wheel
```

The submodule URL is SSH (`git@github.com:...`); without a key on the box, use
`git config --global url."https://github.com/".insteadOf git@github.com:`.

### Correctness first

```bash
# batched encoder vs the banked CPU oracle (CPU only, seconds)
PYTHONPATH=third_party/leapXELA_model:explore \
  systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
  .venv/bin/python tests/test_encoder_oracle.py

# GPU tactile path vs the CPU reference, at a small env count
systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
  .venv/bin/python benchmarks/harness.py --num-envs 4 --tactile splat --verify
```

`harness.py` puts `src/`, `explore/` and `third_party/leapXELA_model` on
`sys.path` itself, so it needs no `PYTHONPATH`; the test's `PYTHONPATH` above
matches its own docstring.

### The sweeps

`scale_sweep.py` runs `harness.py` once per env count in its own
`systemd-run --scope` with a hard `MemoryMax`, so an out-of-memory kills a child
rather than the host. A failure at some N is a result — the ceiling — not a crash.

```bash
# H1 control: bare physics, N = 1 .. 4096
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile none --tag physics_only

# H2: physics + the splat encoder, same ladder
.venv/bin/python benchmarks/scale_sweep.py \
  --max-envs 4096 --mem-cap 60 --tactile splat --tag splat
```

Both write to `benchmarks/results/scale_sweep_<tag>.csv`. Run inside `tmux` so a
dropped SSH connection does not kill the job.

### The figures

```bash
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  .venv/bin/python analysis/plots.py
```

Regenerates all four PNGs in `analysis/plots/` from the two CSVs alone. No number
on any figure is hardcoded — crossover, saturation knee, slowdown and the Amdahl
ceiling are all recomputed at plot time, so re-running the sweeps updates the
titles too. See `analysis/README.md`.

## 8. Repository layout

```
├── README.md                  # this file
├── HYPOTHESES.md              # H1-H6 + scope decisions D1-D8, registered before measuring
├── PROJECT_LOG.md             # durable running record: what was done, decided, learned
├── third_party/
│   ├── leapXelaMjLab/         # submodule: the supervisor's GPU training stack
│   └── leapXELA_model/        # submodule: hand models + the CPU reference encoder
├── explore/                   # model archaeology: body map, taxel injection, oracle fixture
├── src/encoders/taxel_torch.py
├── benchmarks/                # harness.py, scale_sweep.py, results/*.csv
├── tests/test_encoder_oracle.py
└── analysis/                  # plots.py -> plots/*.png
```

Planned and not yet present: `src/obs_terms/` (the mjlab `ObservationTermCfg`),
`src/kernels/{triton,cuda}/`, `benchmarks/contact_budget_sweep.py`,
`profiling/nsight/`.

## 9. Attribution

The hand models, the taxel layout, and the reference encoder
(`VirtualTaxelSensor`, `taxel_layout.py`) are the supervisor's work
([`mohammad200h/leapXELA_model`](https://github.com/mohammad200h/leapXELA_model)),
as is the GPU training stack
([`leapXelaMjLab`](https://github.com/mohammad200h/leapXelaMjLab)). Both are
vendored here as submodules (via forks under `AfthabShiraz/`) and are unmodified.
Much of the recent sensor-modelling and dataset work in those repos is by
`pratik-ingle`. This repository adds the body map,
the taxel injection into the RL model, the benchmark harness, the batched
encoder, and the measurements above.
