# Project 2 — Tactile Observations at Training Scale: Closing the GPU Gap for a Dexterous Hand

**A handoff & teaching document.** Written for a Claude instance that will teach and pair-program this project with the author (Afthab). Read "How to use this document" first — the teaching approach matters as much as the content. This is the **second** of three HPC projects; it assumes the author has done, or is doing, Project 1 (change-aware VLM inference) and has meaningfully progressed in C++.

**Context:** Afthab's supervisor (Hamid, `github.com/mohammad200h`) has built a LeapXELA tactile dexterous hand across several repos. This project fills a specific, verified gap in that body of work. §1.5 maps the terrain — **read it before anything else**, because the gap *is* the project.

---

## 0. How to use this document (instructions for the teaching Claude)

The learner's goal is to **understand parallel-computing performance deeply** by building this — not merely to produce a working sim. Optimise for that.

- **The thesis is "the bottleneck moves."** Once physics is on the GPU, physics stops being the bottleneck and something else dominates. Here we have a *strong prior* about what (tactile observation encoding — §2), and circumstantial evidence for it (§1.5). **That does not excuse skipping the measurement.** Register the hypotheses in writing at Phase 0, then let the profiler confirm or refute them. "I predicted it and measured it" is stronger and more honest than a staged discovery.
- **Measure at every scale.** The core artifact shows how the bottleneck *migrates* from tens to thousands of parallel environments. Benchmark at multiple scales, not just the top end.
- **Do not rebuild what exists.** The supervisor has already built the GPU training stack — twice. Phase 1 is *reproduction*, not construction. This is a gift: it removes weeks of risk and lets the learner spend their time on the performance work, which is the actual point. Push back hard if the learner starts reimplementing his env.
- **This is the C++ project**, but pitch it to where they are. The C++ work is designed so the *same algorithm* is written three times (NumPy → C++/OpenMP → CUDA), which is the highest-value use of their C++ time. If C++ isn't ready, relocate the phase (§5, Phase 4) rather than faking it in Python.
- **Guard the deliverable floor.** Success = "added tactile observations to a GPU-scale RL pipeline, benchmarked across scales, found and fixed the bottleneck, wrote it up." The migration story is upside on a guaranteed systems deliverable. Don't stall chasing drama.
- **This is real internship work with a real user.** The gap in §1.5 is one the supervisor will have to close eventually anyway. Keep him in the loop, ask before assuming, contribute back (§10).
- **Assume:** strong Python/PyTorch/ML, RL familiarity, MuJoCo exposure, C++ in progress, new-ish to GPU performance profiling. Explain profiling and parallel-scaling from first principles; don't over-explain RL.
- **Never report a kernel speedup without a correctness check.** §7 is not optional.

---

## 1. The one-paragraph project summary

The supervisor's LeapXELA hand has **368 taxels** and trains in-hand cube reorientation at **4096 parallel environments on GPU** — but the policy observes **no touch at all**, because the hand's tactile encoders are CPU-only Python and nothing tactile is wired into the training env. *(Precise version, verified 2026-08-13 — do not overclaim this: native MuJoCo `<touch>` sensors **do** run on GPU under Warp, they are simply unwired here and unmeasured at scale. What exists nowhere on GPU is the hardware-faithful 368-taxel **splat** encoder — 3 channels, no extra collision geoms, indices matching real XELA hardware IDs. See §1.6.)* This project closes that gap and treats the closing as a performance investigation. It ports the taxel encoder into the GPU training stack (mjlab / MuJoCo Warp / PyTorch), benchmarks throughput across scales *and* across tactile representations, and demonstrates the bottleneck migrating: at low env counts physics dominates; at thousands of envs, turning contacts into a 368-taxel map dominates, because that encoder is an irregular scatter nobody wrote to run in parallel. It then kills that cost with a custom Triton/CUDA kernel, validated numerically against the supervisor's existing CPU implementation, and trains a tactile policy at scale. The deliverable is a **cost-of-tactile-sensing analysis**: what each way of sensing touch costs you at training scale, and how to get it back.

---

## 1.5 The terrain: what already exists (verified, read this first)

Four repos, checked directly. **Every claim here is from reading the code, not the READMEs.**

| Repo | What it is | Backend | Tactile? |
|---|---|---|---|
| `leapXELA_model` | Hand models + CPU tactile toolkit; 4 sensing implementations | CPU MuJoCo 3.5 | **Yes** |
| `leapXELA` | Cube-reorient env (`reorient.py`), mujoco_playground style | **MJX-JAX**, GPU | No |
| `leapXelaMjLab` | Cube-reorient at 4096 envs, multi-GPU PPO, WandB, hparam sweeps | **MuJoCo Warp** (mjlab + RSL-RL, PyTorch) | **No** |
| `sparsh-skin-sim` | Flex-tactile env + Sparsh-Skin data collection, vectorised env | **CPU**, gymnasium | Yes |

**The gap is the intersection: tactile × GPU-scale. It is empty.**

Supporting evidence, all verifiable:

- `leapXelaMjLab/src/leap_xela_mjlab/tasks/reorient/mdp/observations.py` — the full observation set is `joint_pos`, `joint_pos_error`, `cube_pos_error`, `cube_ori_error`, `last_action`; the asymmetric critic adds `joint_vel`, `fingertip_pos`, `cube_lin_vel`, `cube_ang_vel`. **No taxels.** On a tactile hand.
- `leapXELA_model` contains **no GPU code in any of its 6 branches or 44 commits** — the only `import torch` is CoACD mesh decomposition (`torch>=2.9.1` sits in its `pyproject.toml` next to `coacd`), and every "mjx" occurrence is a filename being written. Its role is model generation + CPU tactile readout.
- `sparsh-skin-sim` is the active tactile work (pushed within the last day) and is **CPU**. A recent commit is *"changed sync in vecenv to async"* — i.e. the supervisor is fighting CPU vectorisation throughput for tactile data collection right now. That is the pain this project removes.
- `leapXELA_model` is a **git submodule** of `leapXelaMjLab` at `src/leap_xela_mjlab/assets/leapXELA_model`. The wiring pattern already exists; follow it.

**What this means for the learner:** the RL, the task, the reward, the domain randomisation, the multi-GPU PPO and the 4096-env scaling are all *done and working*. That is not a loss of scope — it is the difference between "I built a sim" (weak signal, months of work) and "I found and fixed the bottleneck blocking a real research capability" (strong signal, and the actual HPC project). Spend the saved weeks on profiling and kernels.

**Bonus axis, free of charge:** the supervisor built the same task on **both** MJX-JAX and MuJoCo Warp. A backend comparison on identical physics is available with almost no extra work (§9).

**There is a second contributor.** `pratik-ingle` has sent six PRs since 2026-07-31 (#4–#9; **#8 open**) and owns most of the recent sensor-modelling work — the `leapxela` package itself, flex/bubble sensor generation, taxel viewers, taxel ID maps, grasp trajectory export. Contribution counts: `leapXELA_model` 39/5, `sparsh-skin-sim` 10/1, `leapXelaMjLab` 3/0. So the ownership split is **Hamid = sim + hardware + RL, Pratik = sensor modelling + datasets, nobody = performance**. Check his open PRs before touching sensor-model files, and treat him as a collaborator to coordinate with, not an abstraction.

## 1.6 Prior art — the redundancy check (done 2026-08-13, redo before committing)

The §1 claim is only as good as this check. What was verified:

- All non-`main` branches of every repo are old model-generation work (newest, `flex_sensor`, 2026-08-03). `leapXelaMjLab`, `sparsh-skin-sim` and `leapXELA` have **only `main`**, no open PRs, no issues. The only forks are Pratik's.
- **`langxin11/contactile_mjlab`** (created 2026-05-19, last push 2026-05-29, 0 stars) *does* implement tactile observations in mjlab — but on a **Robotiq 2F-85 gripper**, **18 taxels** (3×3 per side), read from builtin `<force>` sensors on sphere geoms. No encoder, no kernel, no scaling analysis. **Not redundant**; useful as prior art for the geom+builtin-sensor representation, and worth citing.
- **MuJoCo Warp already implements native touch sensors on GPU** — `mujoco_warp/_src/sensor.py` L2060 `_sensor_touch`, dispatched `dim=(naconmax, sensor_touch_adr.size)`. This is why the §1 wording had to be corrected. It is also an *opportunity*: representation 3 becomes a near-free GPU baseline to measure against rather than something to build, and that `naconmax × n_sensors` dispatch shape is an unmeasured scaling cost sitting in plain sight (directly relevant to H3).

**Still unresolvable from outside:** unpushed local work. §10 Q1 remains the one question that could invalidate the project — ask it first.

---

## 2. The core idea, explained properly (read before building)

### 2.1 Why "put it on the GPU" isn't the project

Anyone can run many environments in parallel — Warp/MJX do the hard parallel-physics work, and here it is already done. *Using* it is not an HPC signal.

The signal comes from the layer *around* the sim. A training step is not just physics:

1. **Stepping the physics** (MuJoCo Warp, on GPU)
2. **Computing observations** — for a tactile hand, a 368×3 taxel map derived from contacts
3. **Computing rewards**
4. **Moving data** between sim, policy, and back
5. **The policy forward/backward pass**

At low env counts (1) dominates. But physics parallelises well, so as env count climbs, (1) gets cheap *per env* while (2)–(4) may not. **The bottleneck migrates from physics to the pipeline around it.** Finding, explaining, and fixing that is the project.

### 2.2 Why tactile sensing is the ideal stress case (the mechanism)

Three structural facts make this specific rather than generic:

**(a) The taxel encoder is an irregular scatter written in Python.** `leapXELA_model/leapxela/touch_sensor.py` → `VirtualTaxelSensor.update()` loops over `data.ncon` contacts; per contact it calls `mj_contactForce`, finds which hand body was touched, computes Gaussian splat weights over *that body's* taxels, normalises, and scatter-accumulates a 3-vector into a (368, 3) buffer. Per env, in Python, on host. This is textbook "wasn't written to scale," and it is the single reason the GPU stack has no touch.

**(b) Sensing resolution can be bought with geometry — and geometry is charged at the contact budget.** The repo's binary-touchgrid model (`robot_touch_sensor_array_binary_touchgrid_generated_mjx.xml`) adds **361 collision geoms** to get taxel resolution. The splat encoder adds **zero**. Both engines cap contacts statically — `leapXelaMjLab` sets `nconmax=48`, `njmax=120` (**verified** in `env_cfg.py`); the model files were said to set `max_contact_points=30`, `max_geom_pairs=12` (**UNVERIFIED — neither string appears anywhere in `leapXELA_model`; find the real values before citing these**) — so the geom-based route either blows the budget or silently drops contacts. Same sensing goal, radically different cost, for a reason you can derive.

**(c) Backend choice determines what is even possible.** MJX-JAX does not support flex at all — **verified**, `mjx/mujoco/mjx/_src/io.py:315` raises `NotImplementedError('Flex not implemented for JAX backend.')`. MuJoCo Warp does support flex (`mjGEOM_FLEX`, `mjOBJ_FLEX`, `mjEQ_FLEX`, `mjEQ_FLEXSTRAIN`, `nflex`, and flex broadphase/narrowphase/CCD paths in `_src/types.py`); `ELASTICITY` appears in the source, `VERTCOLLIDE` does not appear under that exact name — check the current enum spelling before quoting it. So the highest-fidelity sensor — the flex bubble skin in `leapXela_generated_flex_sensor.xml`, 385 bodies / 1129 joints / 368 connects, currently CPU-only in `sparsh-skin-sim` — is *plausibly* GPU-viable on Warp and definitively not on MJX-JAX. Whether it is viable in practice at scale is an open, worthwhile, unanswered question. Treat it as a stretch result, not an assumption.

Together: **four ways to sense touch, differing by orders of magnitude in cost, for structural reasons.** That is the project.

### 2.3 Hypotheses (write these into the repo before measuring)

> *H1.* With tactile off, physics dominates step time at all reachable env counts, and throughput scales near-linearly until GPU saturation. *(Reproduces the supervisor's existing regime — the control.)*
>
> *H2.* With a naive port of the splat encoder on, the encoder's share of step time grows with env count and crosses over the physics step somewhere in N ∈ [64, 1024], after which throughput is encoder-bound.
>
> *H3.* The geom-based touchgrid degrades throughput even excluding readout cost, because it inflates the static contact budget — its cost appears *inside* the physics step.
>
> *H4 (stretch).* Flex-based tactile is viable on Warp but at a throughput cost large enough to make it impractical for RL at N ≥ 1024.

Crossover points and magnitudes are genuinely unknown; those are the findings. Headline figure: **per-stage step-time breakdown vs. env count, one panel per tactile representation.**

### 2.4 Why this justifies the custom kernel honestly

The profiler indicts the encoder, and the encoder is an irregular gather/scatter — exactly the shape where a hand-written kernel beats a naive framework implementation. You are not picking an operation arbitrarily.

Second piece of luck: **the CPU implementation is a correctness oracle.** `VirtualTaxelSensor` is already validated against grasp tests. Your kernel must reproduce it to tolerance. "200× faster *and* matches the reference to 1e-5" is far stronger than a speedup alone.

### 2.5 Where the C++ lives

One algorithm, four implementations — the pedagogical spine:

| Implementation | Phase | What it teaches |
|---|---|---|
| NumPy/Python (exists, in repo) | 0 | Reference semantics + the oracle |
| Native C++ / MuJoCo C API, then OpenMP | 4 | Memory layout, cache behaviour, parallel decomposition, false sharing, sync cost |
| Triton, then CUDA (torch extension) | 5 | GPU memory hierarchy, coalescing, atomics vs. gather reformulation, occupancy |
| *(optional)* `warp.kernel` | 5 | What a DSL buys and costs vs. hand-written CUDA |

The learner writes the *same* Gaussian-splat encoder at every level of the stack and measures all of them. The sim itself is Python-facing; the C++ is the baseline and the development bed for CUDA. Be explicit about that split.

**Backend note:** mjlab is PyTorch-based. A Triton or CUDA kernel therefore drops in as an ordinary torch extension consuming torch tensors — dramatically simpler than a JAX custom call. This is a large practical reason the plan targets Warp/mjlab rather than MJX-JAX.

---

## 3. Prerequisites & environment

**Hardware**
- One NVIDIA GPU minimum; the supervisor trains at 4096 envs, so match that if possible. Two GPUs unlock his `--gpu-ids` multi-GPU path. Check the UCL Robotics Institute cluster first — likely the cheapest route, and the supervisor already has a working setup to copy.
- **⚠ The current dev box is a DGX Spark (GB10, aarch64) with ~121GB of *unified* memory shared between CPU and GPU — there is no separate VRAM.** An oversized `--num-envs` does not raise `CUDA out of memory`; it exhausts host RAM, the OOM killer starts killing system daemons, and the machine wedges (this happened on 2026-08-13 and took the box down). **Never launch a scaling run uncapped:** `systemd-run --user --scope -p MemoryMax=90G -p MemorySwapMax=0 <cmd>`, inside `tmux`, ramping N and recording peak RSS. Assume **4096 envs is not reachable on the Spark** — treat it as the development/profiling/kernel machine and plan the top-end scaling points (Phase 1 baseline, Phase 6 training) for the cluster. Securing that access is now week-one critical, not optional. Running state and the crash forensics live in `PROJECT_LOG.md`.
- Nsight Systems + Nsight Compute.

**Software**
- `leapXelaMjLab`, running: Python 3.10–3.12, `uv sync`. Pulls `mjlab` from git, plus `mujoco>=3.3`, `torch`, RSL-RL. Clone **with submodules** — the hand model is one.
- `leapXELA_model` (CPU, poetry, `mujoco==3.5.0`) for the reference encoder and the oracle fixtures.
- WandB account (his training logs to it by default).
- CUDA toolkit + `nvcc`; Triton (ships with torch).
- MuJoCo C/C++ headers for Phase 4.
- *Optional:* `leapXELA` (MJX-JAX) for the backend comparison.

**Version friction to expect:** three repos, three dependency managers (uv, poetry, poetry), and two MuJoCo versions (3.3+ vs 3.5.0). Keep them in separate environments and do not try to unify them. Budget for this.

**Skills assumed:** Python/PyTorch, RL familiarity, MuJoCo. **Skills built:** parallel-scaling analysis, GPU profiling at scale, irregular-kernel writing, native-MuJoCo C++.

---

## 4. Scope decisions (lock these in Phase 0)

Keep scope *ruthlessly* narrow — the value is depth of performance analysis, not breadth of robotics.

- **Task, robot, RL algorithm, reward: inherited, unmodified.** `Mjlab-LeapXELA-Cube-Reorient` with RSL-RL PPO (actor/critic 512-256-128 ELU, 24 steps/env, adaptive KL). `sim_dt=0.01`, `decimation=5` → 20 Hz control. **Do not touch any of this.** Changing the task invalidates comparison against his baseline, which is your control group.
- **The one thing you change is the observation space** — adding tactile — plus whatever the profiler forces.
- **Scaling range, fixed for the whole project:** N ∈ {16, 64, 256, 1024, 4096}, extended to 8192/16384 if memory allows.
- **Tactile representations to compare** (three is the floor, four is upside):
  1. **None** — his current observation set. The control.
  2. **Splat encoder** — the `VirtualTaxelSensor` algorithm, 368 taxels × 3 channels, no extra geoms. *The primary kernel target.*
  3. **Native `touch` sensors** — `robot_touch_sensor_array_mjx_generated_model.xml` has 421 `<touch>` sensors on sites. Cheap to try; good comparison.
  4. **Binary touchgrid** — 361 collision geoms. Expect it to be brutal; that's H3.
  - *(Flex bubble skin — H4, stretch. Warp supports flex in principle; whether it survives at scale is the question.)*
- **Backend: mjlab / MuJoCo Warp.** MJX-JAX is a comparison data point (§9), not a second implementation.
- **Integration point:** mjlab's manager API means tactile enters as a new `ObservationTermCfg` in `tasks/reorient/mdp/observations.py`. That is a clean, small, reviewable diff. Keep it that way.

---

## 5. Build order — phases, each ending in a measurement

Ordering rule: **every phase produces a number; no phase optimises what the previous phase didn't prove was slow.**

### Phase 0 — Orientation, reproduction, harness & hypothesis lock *(first, always)*

**Goal:** understand both stacks from the inside, build the ruler, commit to the hypotheses.

- Get `leapXELA_model` running on CPU. Reproduce `reading_contact.py` and `grasp_touch_test.py` (closes fingers on an object, logs all 368 taxels to `.npz` with heatmap renders). If those run, the tactile side is real to you.
- **Read `leapxela/touch_sensor.py` and `leapxela/taxel_layout.py` until you can explain the splat algorithm on a whiteboard.** Everything downstream is this function.
- Get `leapXelaMjLab` running: `uv sync`, then `play.py` with `--agent random`. Then a short `train.py` run at low `--num-envs` to confirm the GPU path works end to end.
- Build a **repeatable benchmark harness**: warm up, time many steps, report **environment-steps/sec** and wall-clock per PPO iteration, plus GPU utilisation. Handle CUDA async correctly — `torch.cuda.synchronize()` or you measure nothing.
- **Write H1–H4 (§2.3) into the repo, dated, before measuring.**
- Lock §4.

**Ends with:** both stacks reproduced, throughput harness, baseline number, hypotheses committed. *Teaching note: no optimisation. Define the ruler and state the prediction first.*

### Phase 1 — Reproduce the supervisor's scale baseline *(fast, and it de-risks everything)*

**Goal:** a known-good tactile-free training run at full scale, as your control.

- Train `Mjlab-LeapXELA-Cube-Reorient` at 4096 envs to a non-trivial success rate — or, if compute is tight, reproduce his WandB curve for a fixed iteration budget and confirm you match.
- Record throughput and time-to-target-return. **This is the number every later result is compared against.**
- Sanity-check your environment against his: same seed, same config, similar curve. Divergence here is a setup bug, and finding it now costs hours instead of weeks.

**Ends with:** a reproduced 4096-env baseline + its throughput. *Teaching note: this phase is short by design. Resist the urge to "improve" anything — you are calibrating, not building.*

### Phase 2 — The scale sweep & per-stage profiler (establish the physics-dominant regime)

**Goal:** the instrument the whole thesis rests on, plus the H1 result.

- Sweep N ∈ {16 … 4096} with tactile **off**. Plot throughput vs. env count. Where does it stop scaling, and why — compute saturation or memory?
- Build the **per-stage timing breakdown**: physics step / observation / reward / policy forward+backward / any host-device traffic. Nsight Systems plus torch profiler. Getting an *honest* breakdown out of a mjlab step is itself a real skill — budget time for it, and beware CUDA async when attributing time.
- Sweep the contact budget (`nconmax=48`, `njmax=120` in `env_cfg.py`; `max_contact_points`, `max_geom_pairs` in the model XML) at fixed N. How much of physics cost is contact-budget-driven? This calibration is what lets you interpret H3.
- **Test H1.** Report it however it comes out.

**Ends with:** throughput-vs-N curve, per-stage breakdown at every N, contact-budget sensitivity, H1 resolved. *Teaching note: this is the ruler for everything after. Do not rush it.*

### Phase 3 — Turn tactile on and watch it break (the migration figure)

**Goal:** the headline result.

- Implement the **splat encoder** as an mjlab observation term — a faithful, *unoptimised* PyTorch port of `VirtualTaxelSensor.update()`. Faithful first: it must match the CPU reference (§7) before it is allowed to be slow-and-interesting.
- Re-run the full sweep with it on. Per-stage breakdown at each N. **Where does the encoder overtake the physics step?** That crossover is the money number.
- Repeat for the **native `touch` array** (421 sensors) and the **binary touchgrid** (361 geoms). For the touchgrid, separate cost *inside* the physics step (inflated contact budget, possibly dropped contacts) from readout cost — that's H3. Check whether the contact budget is silently truncating; a sensor that drops contacts is wrong, not just slow.
- Measure the **memory** cost too: 368×3×N, plus inflated contact arrays. Does max feasible N drop? By how much?
- If time allows, attempt flex on Warp (H4) and report what happens — including "it doesn't fit," which is a legitimate result.

**Ends with:** the **cost-of-tactile-sensing figure** — per-stage breakdown vs. N, one panel per representation — plus H2/H3 resolved. *Teaching note: this is where the thesis becomes visible in data. Make the learner articulate what they see before explaining it.*

### Phase 4 — The C++ interlude: the splat encoder in C++ and OpenMP *(the one relocatable phase)*

**Goal:** systems competence, an honest CPU baseline, and the development bed for the CUDA kernel.

- Reimplement the encoder in **native C++** against MuJoCo's C API, reading `mjData.contact` directly. Benchmark against the Python original, single env. Expect a large interpreter win — **first measured C++ result.**
- Parallelise across environments with `std::thread` or OpenMP. Measure **throughput vs. core count.** Where does it plateau, and *why*? Memory bandwidth? Sync overhead? False sharing on the accumulation buffer?
- **The false-sharing discussion here is the direct precursor to the atomics-vs-gather decision in Phase 5.** Spend real time on it; it is the conceptual bridge.
- Bonus relevance: this is also directly useful to `sparsh-skin-sim`, which is CPU and currently fighting vectorised-env throughput.

**Ends with:** C++ and OpenMP encoders, a CPU core-scaling curve, and an explanation of the plateau. *Teaching note: this is the one phase whose position is flexible. If C++ readiness lags, run Phase 5 first and return. Do not fake C++ in Python.*

### Phase 5 — The GPU kernel (Triton, then CUDA)

**Goal:** the HPC payoff — kill the cost Phase 3 proved dominant.

- **Profile at high N with Nsight Compute** before writing anything. Latency-bound, bandwidth-bound, or occupancy-bound? Let the profiler set the target.
- **Design the kernel properly — the teaching centrepiece.** The reference is a *scatter*: loop contacts, splat into taxels, atomically accumulate. The better GPU formulation is usually the *transpose* — one block per (env, taxel), looping that env's contacts, accumulating in registers, **no atomics**. With 368 taxels and a contact budget of ~48, there is ample parallelism either way, so the gather form wins on memory traffic and determinism. Make the learner derive this rather than being told.
  - Complication to work *through*, not around: the reference **normalises weights per contact** (`weights /= weight_sum`) — a reduction over a contact's taxels that must complete before accumulation. The gather form needs either a cheap pass-1 computing per-contact `weight_sum`, or a fused two-stage kernel. Choosing between them, with measurements, is exactly the judgement this project demonstrates.
  - The static contact budget that *hurts* in Phase 2 **helps here**: contacts arrive as a fixed `(N_env, nconmax, …)` tensor, so the kernel has static shapes and no ragged handling. Worth calling out in the write-up.
- **Triton first** (fast iteration, reuses Project 1 skills), **then CUDA** as a torch extension for depth: coalescing on site position/orientation reads, shared memory for the per-body taxel slice, occupancy tuning, roofline.
- *Optional:* a `warp.kernel` version, since Warp is already a dependency — a clean "what does the DSL cost you" comparison.
- **Validate against the CPU oracle at every step (§7).** A fast wrong kernel is worth zero.
- Benchmark **end-to-end before/after**. Report Amdahl honestly: if the encoder is 60% of step time, the ceiling is 2.5×. **State the ceiling before reporting the achieved number.**

**Ends with:** validated Triton + CUDA encoders, roofline, measured end-to-end speedup with the Amdahl bound stated. *Teaching note: strongest single artifact. Keep the "profiler chose this, not me" link explicit.*

### Phase 6 — Tactile RL end to end (the payoff)

**Goal:** show the optimisation bought something a researcher cares about.

- Train the reorientation policy **with tactile observations** at 4096 envs — the thing that could not be done before this project.
- Report **time-to-target-success-rate**: tactile-naive (Phase 1 control) vs. tactile-with-naive-encoder vs. tactile-with-kernel. This converts throughput into research velocity.
- Re-profile with the full loop present; the policy forward/backward and plumbing may shift the bottleneck *again*. A second migration is a great result, not a failure.
- **Do not over-claim on the RL outcome.** Whether tactile observations *improve the policy* is a research question with its own confounds, and it is not what this project is measuring. If tactile training is now feasible where it wasn't, that is the deliverable. Any policy-quality result is a bonus, reported cautiously.

**Ends with:** a trained tactile policy at scale + wall-clock training improvement + final bottleneck attribution. *Teaching note: this is the sentence the supervisor and any interviewer will remember.*

### Phase 7 — Consolidate & write up

- Assemble the narrative: gap identified → baseline reproduced → scale sweep (physics-bound) → tactile on (encoder-bound) → C++/OpenMP baselines → GPU kernel → tactile training unlocked.
- Structure the repo as a mini-paper (§6). Include the Phase-0 hypothesis register **with resolutions**, including any that were wrong. That honesty is a feature.
- Figures: throughput-vs-N; cost-of-tactile-sensing panel; the migration figure; CPU core-scaling; Nsight screenshots; roofline; before/after training curves.

---

## 6. Repository structure (target)

Your repo is separate from the supervisor's. Follow his own pattern — he vendors `leapXELA_model` as a submodule — so provenance stays unambiguous and his work is clearly attributed.

```
tactile-hand-rl-scaling/
├── README.md                   # mini-paper: gap, hypotheses, method, results
├── HYPOTHESES.md               # Phase 0 register + dated resolutions
├── third_party/
│   ├── leapXelaMjLab/          # submodule: supervisor's GPU training stack
│   └── leapXELA_model/         # submodule: models + CPU reference encoder
├── benchmarks/
│   ├── harness.py              # env-steps/sec + per-iteration wall clock
│   ├── scale_sweep.py          # {16..4096} × {none, splat, touch, touchgrid}
│   ├── contact_budget_sweep.py # nconmax / njmax / max_contact_points
│   └── results/                # committed CSV/JSON for every run
├── analysis/
│   ├── bottleneck_migration.py
│   ├── validate_encoder.py     # §7 oracle comparison
│   └── plots/
├── cpp/
│   ├── taxel_encoder/          # Phase 4 native MuJoCo C API
│   ├── taxel_encoder_omp/      # Phase 4 OpenMP across envs
│   └── CMakeLists.txt
├── src/
│   ├── obs_terms/              # mjlab ObservationTermCfg for each tactile mode
│   ├── encoders/
│   │   ├── reference_numpy.py  # faithful port of VirtualTaxelSensor (oracle)
│   │   └── naive_torch.py      # Phase 3 unoptimised GPU version
│   └── kernels/
│       ├── triton/
│       └── cuda/               # torch extension
└── profiling/nsight/           # captured profiles + written notes
```

**README must contain:** the gap (§1.5) in three sentences; the hypotheses and how they resolved; throughput-vs-N curves; the cost-of-tactile-sensing figure; kernel before/after with the Amdahl ceiling stated; the roofline; the encoder validation error; and exact repro instructions including CUDA/torch/mjlab versions.

---

## 7. Correctness & validation (non-negotiable)

Every encoder implementation — naive torch, C++, OpenMP, Triton, CUDA — must be checked against the **CPU reference** (`VirtualTaxelSensor.update()`) before any timing is reported.

- **Build the oracle fixtures first.** `grasp_touch_test.py` already dumps an `.npz` of all 368 taxels through a scripted grasp. Save contacts-in → taxels-out pairs across a range of contact counts, including `ncon = 0` and a heavy multi-finger grasp.
- **Compare every implementation** on max absolute and max relative error per channel. Agree a tolerance up front (float32 accumulation-order differences are expected; ~1e-5 relative is reasonable) and **report the achieved error next to every speedup in the README.**
- **Edge cases that break naive kernels:** zero contacts; `weight_sum == 0` (the reference `continue`s — your kernel must emit zeros, not NaNs); contacts on bodies with no taxels; the contact budget being hit.
- **Cross-engine check:** MuJoCo Warp and CPU MuJoCo may not produce identical contacts for the same state. Establish how closely they agree *before* blaming your encoder for a mismatch. This distinction has burned people.
- **Non-determinism is a finding, not a nuisance.** If atomics make results vary run to run, quantify the spread — a good argument for the gather formulation and a good paragraph in the write-up.

---

## 8. Definition of done (the deliverable floor)

1. **Tactile observations run in the GPU training stack** at 4096 envs — a capability that did not exist before.
2. **Throughput benchmarked across scales**, with per-stage breakdowns, for at least three tactile representations.
3. The **bottleneck migration is demonstrated with data**, and each Phase-0 hypothesis is reported confirmed or refuted.
4. At least **one custom kernel** (Triton; ideally also CUDA) accelerates the encoder, **validated against the CPU reference**, with measured end-to-end improvement and profiling that justifies it.
5. It is **public, clean, reproducible**, with a paper-quality README, and the supervisor's work properly attributed.

Upside: C++/OpenMP baselines (Phase 4), deep CUDA optimisation, all four representations, flex-on-Warp (H4), the MJX-vs-Warp comparison, a Warp-kernel version, larger N. **Ship the floor, then reach.**

---

## 9. Risks & how to handle them

- **Environment setup across three repos eats week one.** Three dependency managers, two MuJoCo versions, mjlab pulled from git `main` (a moving target — pin the commit). Get one working setup, document it, don't upgrade mid-project.
- **mjlab is young and moves fast.** Pin the commit hash. If an upstream change breaks you mid-project, pin and move on; don't chase `main`. **This risk is already live:** `leapXelaMjLab/pyproject.toml` declares `mjlab = { git = "https://github.com/mujocolab/mjlab", branch = "main" }` — an unpinned moving branch (mjlab is ~2.8k stars and actively developed). Pin it in your fork on day one and record the hash.
- **The submodule URL is SSH.** `leapXelaMjLab/.gitmodules` uses `git@github.com:mohammad200h/leapXELA_model.git`, so `--recurse-submodules` fails without a working SSH key on the box. Either add a key or set `git config --global url."https://github.com/".insteadOf git@github.com:`.
- **The supervisor ships changes under you.** `sparsh-skin-sim` was pushed within the last day. Pin submodule commits, and check in with him before starting each phase so you don't collide.
- **Reproducing his baseline fails.** Most likely a config/seed/version difference, not a real discrepancy. Ask him — he has WandB logs of working runs. Do not proceed to Phase 2 on an unreproduced baseline.
- **Warp/CPU contact discrepancies confound encoder validation.** See §7. Separate engine differences from your bugs *before* debugging the kernel.
- **The bottleneck doesn't migrate as predicted.** Possible — a batched torch encoder may vectorise better than expected. That is a *reportable result against a registered hypothesis*, which is the honest outcome. Pivot the target to whatever the profiler indicts; the systems deliverable stands regardless.
- **Amdahl caps the headline number.** State the ceiling before reporting the result. A well-explained 1.6× beats an unexplained 10×.
- **GPU access limits high N.** The thesis needs scale, and the supervisor already trains at 4096. Secure equivalent access in week one.
- **Scope creep into RL or task design.** The task, reward, and algorithm are inherited and frozen. Any novelty belongs in the performance layer.
- **Kernel work stalls.** Triton is the floor; CUDA is upside.
- **MJX-JAX comparison temptation.** He has both backends, so the comparison is cheap and interesting — but it is a *side result*, worth at most a section. It is not a second implementation track.

---

## 10. Working with the supervisor (this is internship work too)

There is a clean, non-duplicative contribution story here, but confirm it before building.

**Ask first — these change the plan if the answers surprise:**
- "Is anyone already putting tactile observations into `leapXelaMjLab`?" *(The gap looks real, but he may have unpushed work — this is the one question that could invalidate the project, so ask it first.)*
- Which tactile representation does he consider canonical for RL — splat, native `touch` array, or flex/Sparsh-Skin? That decides the primary kernel target.
- Is mjlab/Warp the settled direction, or is MJX-JAX still live? *(The evidence says Warp, but confirm before committing.)*
- Are `nconmax=48` / `njmax=120` tuned or placeholders?
- Where does `sparsh-skin-sim` fit — is tactile RL meant to run there, or in mjlab?
- Has he tried the **native `<touch>` sensor path** in mjlab? Warp supports it on GPU already (§1.6) — if he hasn't, wiring it up is a cheap early win and gives the baseline everything else is measured against.
- **How does Pratik's work relate?** (§1.5) He owns the sensor-modelling lane and has an open PR (#8). Confirm the split, and ask whether the taxel ID maps in #8 are the ones to build against.
- Is he happy for a public repo, and how does he want attribution?

**Offer back:** GPU tactile observations in his training stack, a validated fast taxel encoder (GPU *and* multicore C++ — the latter directly helps `sparsh-skin-sim`'s vectorised-env problem), and a quantitative answer to "what does each tactile representation cost us at training scale?" — a question he will have to answer eventually anyway.

---

## 11. What the learner should be able to say afterward (the interview story)

> *"My lab had a tactile dexterous hand — 368 taxels — training in-hand cube reorientation at 4096 parallel environments on GPU. But the policy couldn't feel anything. The hand's tactile encoder was CPU-only Python: an irregular scatter that ran one environment at a time, so the hardware-faithful taxel map had never made it into the GPU training stack. I profiled the pipeline across scales and showed why. At small env counts the physics dominates; past about N the encoder that turns contacts into a taxel map takes over, because it's an irregular scatter nobody wrote to run in parallel. I also measured the three different ways of sensing touch and showed they differ by orders of magnitude for structural reasons — one buys resolution with 361 extra collision geoms and pays for it inside the contact budget. Then I rewrote the encoder as a CUDA kernel, reformulated from a scatter to a gather so it needed no atomics, validated it to 1e-5 against the existing CPU implementation, and trained the first tactile policy at full scale."*

That is the portfolio signature — **don't guess where the time goes; predict, measure, explain the mechanism, fix the real thing, report honestly** — applied in the parallel-scaling regime, on a real research blocker, complementing Project 1's kernel-level version of the same instinct.

---

## 12. How this relates to the other two projects

- **Project 1 (VLM inference)** — *kernel-level, single-GPU, inference optimisation*: "make one op fast by exploiting structure."
- **Project 2 (this one)** — *parallel-throughput, scaling, simulation*: "make thousands of things run efficiently and find the bottleneck that emerges at scale."
- **Project 3 (distributed / multi-GPU)** — *scale-across-devices*: "communication is the enemy; scaling efficiency is a fight." *(Note: `leapXelaMjLab` already has a multi-GPU path via torchrunx — a natural on-ramp.)*

Together they cover kernel → parallel → distributed, spanning ML-systems and robotics-sim, unified by one signature: *measure, find the non-obvious bottleneck, explain why it exists, fix the real thing, report honestly.*
