# Project Log — Tactile Observations at Training Scale

**Purpose:** durable running record of what we've done, decided, and learned, so no context is lost between sessions. Claude updates this at intervals during work. Newest session at the top of §2.

**Companion docs:** `project2_parallel_rl_sim_pipeline.md` is the *plan* (stable). This is the *state* (changes constantly). If the two disagree, this file is what actually happened.

---

## 1. Standing facts (verified, slow-changing)

### 1.1 Hardware — read this before launching anything

The dev box `spark-cefe` is an **NVIDIA DGX Spark**, not a datacenter DGX:

| | |
|---|---|
| Arch | `aarch64`, GB10 |
| Memory | ~121GB usable of 128GB **unified LPDDR5X, shared CPU+GPU** |
| Access | remote SSH (from `100.118.252.19`) |
| Kernel | `6.17.0-1029-nvidia` |

**There is no separate VRAM.** `nvidia-smi --query-gpu=memory.total` returns `[N/A]`. An oversized `num_envs` does not raise a clean `CUDA out of memory` — it exhausts host RAM, the OOM killer starts killing system daemons, and the machine wedges.

**Confirmed crash, 2026-08-13:** OOM kills of `wireplumber` (18:49:37) and `pipewire` (18:51:31), `sshd` broken pipe at 18:51:47, journal silent thereafter, reboot at 19:10:44. No XID/NVRM errors — GPU hardware was never at fault. The Aug 10 boot also ended uncleanly, so this has likely happened more than once.

**Rule going forward — never launch a scaling run uncapped:**
```bash
systemd-run --user --scope -p MemoryMax=90G -p MemorySwapMax=0 \
  .venv/bin/python <script> --num-envs N
```
Run inside `tmux` so a dropped SSH doesn't kill the job. Ramp N and record peak RSS rather than jumping to the top of the range.

### 1.1b Headless rendering — solved, use EGL

Work is over SSH with no display (`DISPLAY` unset, no Wayland). **Interactive viewers do not work** — `mujoco.viewer.launch_passive()` will fail, so `reading_contact.py` needs adapting before it runs here.

**Offscreen rendering does work and is GPU-accelerated.** `libEGL_nvidia.so.580.173.02` + `10_nvidia.json` are installed; verified 2026-08-13 by rendering a test scene (120×160, 191 distinct colours — real output, not a blank buffer).

```python
import os; os.environ["MUJOCO_GL"] = "egl"   # MUST precede `import mujoco`
import mujoco
with mujoco.Renderer(model, height=480, width=640) as r:
    r.update_scene(data); frame = r.render()   # (H, W, 3) uint8
```

View saved PNG/MP4 inline through VS Code Remote SSH. `grasp_touch_test.py` already writes npz + PNG heatmaps + MP4, so it suits this workflow as-is. **Missing from the venv:** pillow / imageio / matplotlib — nothing can write an image file yet.

### 1.2 Local state

- Repo `/home/afthabshiraz/Mujoco`, branch `main`, **zero commits**. Untracked: the plan doc, `.venv/`, this log.
- `.venv` is a bare install (`mujoco 3.11.0`, `mujoco_warp 3.11.0`, `cuda-toolkit 13.0.3`, torch, numpy 2.5.2) — **not** `leapXelaMjLab`'s `uv sync`.
- None of the supervisor's repos are cloned locally yet.
- Position in the plan: **before Phase 0.**

### 1.3 Hamid's repo landscape (verified via `gh` API, 2026-08-13)

`github.com/mohammad200h` — the four that matter, plus context:

| Repo | Role | Backend | Tactile | Last push | Commits |
|---|---|---|---|---|---|
| `leapXELA_model` | Hand models + CPU tactile toolkit; the reference encoder | CPU MuJoCo 3.5 | **Yes** | 2026-08-12 | 44 |
| `leapXelaMjLab` | Cube-reorient, 4096 envs, multi-GPU PPO | MuJoCo Warp (mjlab + RSL-RL, torch) | **No** | 2026-07-28 | 3 |
| `sparsh-skin-sim` | Flex-tactile env + Sparsh-Skin data collection | **CPU**, gymnasium | Yes | **2026-08-13** | 11 |
| `leapXELA` | Earlier cube-reorient env | MJX-JAX | No | 2026-04-10 | — |
| `LeapXELA_Hardware_ws` | Real-hardware workspace | C++ | — | 2026-08-07 | — |
| `leapXela_finger_ws` | Single index finger | Python | — | 2026-07-03 | — |

`leapXELA_model` branches: `main`, `cleanup`, `collision_model`, `fingertip`, `flex_sensor`, `sensor_allocation`.

**The gap (tactile × GPU-scale) is confirmed empty.** Read `leapXelaMjLab/.../mdp/observations.py` directly: the terms are `joint_pos_abs`, `joint_vel_abs`, `joint_pos_error_from_command`, `cube_pos_error_from_palm`, `cube_ori_error_mat`, `cube_lin_vel`, `cube_ang_vel`, `fingertip_positions_rel_palm`. **No taxel term of any kind**, on a tactile hand.

`env_cfg.py` confirms the plan's numbers: `nconmax=48`, `njmax=120`, `ls_iterations=8`, `decimation=5` (`ctrl_dt=0.05`, `sim_dt=0.01` → 20 Hz).

**Gotcha:** `leapXelaMjLab/.gitmodules` points at the submodule over **SSH** (`git@github.com:mohammad200h/leapXELA_model.git`). Cloning needs a working SSH key on the Spark, or a `url.https://github.com/.insteadOf` rewrite.

### 1.4 There is a second contributor — this changes the contribution strategy

`pratik-ingle` is actively sending PRs and owns most of the recent tactile-model work:

| PR | Title | State |
|---|---|---|
| #9 | Add offline LeapXELA grasp action trajectory exporter | merged 08-10 |
| #8 | Add multi-object grasp dataset tools and taxel ID maps | **OPEN** since 08-07 |
| #7 | Align flex taxel frames and raise solver budget | merged 08-04 |
| #6 | Add leapxela package with taxel layout and pointcloud assets | merged 08-03 |
| #5 | Add flex/bubble sensor generation and taxel viewers | merged 08-03 |
| #4 | Add flexcomp tactile sensor model and taxel visualization tools | merged 07-31 |

Contribution counts: `leapXELA_model` — mohammad200h 39, pratik-ingle 5. `sparsh-skin-sim` — mohammad200h 10, pratik-ingle 1. `leapXelaMjLab` — mohammad200h 3 (solo).

**Read:** Pratik owns *sensor modelling and datasets*. Hamid owns *sim + hardware + RL*. **Nobody owns performance, GPU porting, or kernels** — `leapXelaMjLab` is untouched by anyone else and is the least-developed repo (3 commits). That lane is genuinely open, which is the strongest available argument that this project doesn't duplicate anyone.

### 1.4b Redundancy check — prior art beyond Hamid's account (2026-08-13)

Checked because "is this redundant?" deserves a real answer, not an assumption.

**Ruled out as redundant:** all non-`main` branches of every repo (the five extra `leapXELA_model` branches are old model-generation work, newest `flex_sensor` at 2026-08-03); `leapXelaMjLab`, `sparsh-skin-sim`, `leapXELA` each have **only `main`** and no open PRs or issues; the only forks are Pratik's.

**Two real pieces of prior art found:**

**(a) `langxin11/contactile_mjlab`** — tactile observations in mjlab, created 2026-05-19, last push 2026-05-29, 0 stars, Chinese design docs. **Not redundant:** Robotiq 2F-85 gripper (not LeapXELA), **18 taxels** (3×3 per side, not 368), and it reads **builtin `<force>` sensors on sphere geoms** — no encoder, no kernel, no scaling work. Its `mdp/observations.py` is a thin `env.scene[name].data` read with fixed scale factors. *Useful as:* an existence proof for the geom+builtin-sensor representation in mjlab, and a citable prior for one of the four representations.

**(b) MuJoCo Warp already implements touch sensors — important.** `mujoco_warp/_src/sensor.py` L2060 `_sensor_touch`, dispatched `dim=(d.naconmax, m.sensor_touch_adr.size)`. **Native MuJoCo `<touch>` sensors run on GPU today.**

**Consequence — correct the project's framing.** The plan's line *"every tactile implementation is CPU-only Python / touch has never made it into the GPU training stack"* is **too strong**; do not say it to Hamid in that form. Accurate version:

> The hardware-faithful 368-taxel **splat** encoder (3 channels, no extra geoms, indices matching real XELA IDs) does not exist on GPU anywhere. Native `touch` sensors *do* work on GPU but nobody has wired them into `leapXelaMjLab` or measured them at 421 sensors × thousands of envs. Nobody anywhere has compared what these representations cost at training scale.

This **strengthens** the project: representation #3 becomes a near-free working GPU baseline to measure against rather than something to build, and the `naconmax × n_sensors` dispatch shape of `_sensor_touch` is an unmeasured scaling cost sitting in plain sight — directly relevant to H3.

### 1.5 The reference encoder — `leapxela/touch_sensor.py`

Read in full. `VirtualTaxelSensor.update()` is the algorithm every later implementation must reproduce:

1. Loop `con_idx in range(data.ncon)` — **Python-level, per contact, per env, on host**.
2. Filter to object↔taxel-body contacts via `geom_bodyid`; sign-flip force if the hand is geom1.
3. `mj.mj_contactForce(model, data, con_idx, force6)`; `force_world = contact.frame.reshape(3,3).T @ force6[:3]`.
4. Gaussian splat over *that body's* taxels: transform `contact.pos - site_pos` into each taxel frame (`einsum("kji,kj->ki")`), weight `exp(-dist_sq/(2σ²))` on the tangential plane, zero beyond `kernel_cutoff` in-plane **and** in `|local_z|` (keeps the splat on the contacted patch).
5. **`weights /= weight_sum`** — per-contact reduction that must complete before accumulation. `if weight_sum <= 0.0: continue`.
6. `out[taxel_idx] += weights[:,None] * force_local` — the scatter-accumulate.
7. Post: `out[:,2] *= -1.0` (compression positive), optional per-taxel gain and Gaussian noise, `.astype(np.float32)`.

**Facts that matter for the kernel and for validation:**
- `out` accumulates in **float64** and is cast to float32 only at return. A float32 GPU kernel will differ from the oracle for this reason alone — set the tolerance knowing this, and consider an fp64-accumulate variant when quantifying.
- Within a single contact the taxel indices are unique, but **different contacts on the same body write the same taxels** → the scatter form needs atomics across the contact loop. This is precisely why the gather reformulation (one block per `(env, taxel)`, loop that env's contacts) is the right GPU shape.
- The per-contact `weight_sum` normalisation is the thing that makes the gather form non-trivial: it's a reduction over a *contact's* taxels, but the gather form is indexed by taxel. Needs a cheap pass-1 or a fused two-stage kernel. **This choice, measured, is the core judgement call of Phase 5.**
- `mj_contactForce` has no direct MuJoCo Warp equivalent — the Warp port must derive contact force from the efc/constraint arrays. Expect this to be a real piece of work and a likely source of oracle mismatch. Separate it from encoder bugs (plan §7).
- Edge cases to test: `ncon == 0`; `weight_sum == 0` (must emit zeros, not NaN); contacts on bodies with no taxels; contact budget saturated.

`leapxela/taxel_layout.py`: `N_TAXELS = 368`, `PACK_SHAPE = (26, 31)`. Flat taxel order **equals hardware XELA IDs 0..367**, so sim datasets are index-compatible with real hardware streams. 18 patches (11× 4×4 finger, 3× 4×6 palm, 4 fingertip).

### 1.5b ⚠ The GPU stack's hand model has no taxels in it (found 2026-08-13, on first clone)

Not mentioned anywhere in the plan doc, and it is the **first real engineering task**.

**Three separate problems, stacked:**

1. **mjlab loads a model with zero taxel sites.** `robots/leap_xela.py:get_hand_xml()` loads `leapXela_generated_mjx_{Box,CoACD}.xml`. That file has **71 sites, none of them taxels** — on both the pinned version *and* current `main`.
2. **The taxel sites don't exist in any static XML.** They are injected **at runtime** by `leapxela/scene_builder.py:build_scene_xml()`, which parses a *different* base model (`leapxela/leapXela_pointcloud/robot.xml`) with ElementTree and adds one `<site>` per taxel (name, pos, quat, `size=0.0012`, `group=4`). Hence the error string in `touch_sensor.py`: *"Scene model is missing taxel sites; build it with build_scene_xml"*.
3. **mjlab pins an old model.** `third_party/leapXelaMjLab` pins `leapXELA_model` at **`4e8003f` (2026-06-11)** — **13 commits behind `main`**, and *predating the `leapxela` package entirely* (added 2026-08-03 in PR #6). The pinned tree has no `leapxela/` directory at all.

**So the CPU tactile toolkit and the GPU training stack use different hand models, and the GPU one has never had taxels in it.**

**4. The root cause — two independent model lineages (found 2026-08-13).** Verified by compiling both and diffing body names:

| | **Lineage A — RL/GPU** | **Lineage B — tactile/CPU** |
|---|---|---|
| Source | `leapXela_base_model.xml` | `leapxela/leapXela_pointcloud/robot.xml` |
| Generator | `simplify_model_for_mjx.py` → `leapXela_generated_mjx_{Box,CoACD}.xml` | from ROS `xela_description` |
| Consumers | `leapXelaMjLab` (training) | `taxel_layout.py`, `touch_sensor.py`, `scene_builder.py` |
| Body names | `palm`, `if_bs/px/md/ds`, `mf_*`, `rf_*`, `th_mp/bs/px/ds` | `leap_hand_xela_back_cover`, `finger{,_2,_3}`, `p2/p3/p4_unified*`, `thumb`, `thp*_unified` |

**Body-name overlap between the layout's 12 required bodies and mjlab's model: 0 of 12. Against `robot.xml`: 12 of 12.**

So the 13-line site injection **cannot be ported as-is** — `bodies[e.body]` raises `KeyError` on every entry. The taxel layout was authored against a hand model the GPU stack has never loaded.

**The mapping is semantically obvious but geometrically unproven.** `leap_hand_xela_back_cover`→`palm`, `finger`→`if_ds`, `p2_unified`→`if_md`, and so on — 12 layout bodies onto palm + 4 fingers × 3 links. **But `TaxelEntry.pos/quat` are expressed in *Lineage B* body frames.** A correct name map is not sufficient: if the two exports place link frame origins or axes differently, every taxel lands in the wrong place, and the encoder will produce numbers that look plausible and are wrong. Two independent CAD exports have no reason to agree.

**Recommended approach:** (i) ask Hamid — he built both lineages and may already know they correspond, or have the transform; (ii) derive the name map and **validate geometrically** — render the injected sites and confirm they sit on the skin, and check each site's distance to the nearest mesh surface numerically; (iii) only then port the injection. Do **not** skip (ii).

Rejected alternatives: making mjlab load Lineage B (large change; `robot.xml` needs the compatibility fixups `scene_builder.py` applies), and regenerating Lineage A with taxels (needs taxel positions in Lineage A geometry — same unsolved problem).

**This is a strong candidate for the first real contribution:** it blocks tactile-on-GPU entirely, it's tractable, and the artifact — a validated body map plus taxel sites in the mjx model — is reusable by Hamid, Pratik and `sparsh-skin-sim` alike.

**Consequence:** before any encoder work, the 368 taxel sites must exist in the model mjlab loads. That is model plumbing, not kernel work, and it is Phase 0/1. Options to weigh: (a) bump the submodule to current `main` and port `build_scene_xml`'s site injection into the mjlab spec pipeline (mjlab uses `MjSpec`, which is *newer and nicer* than the ElementTree hack `scene_builder.py` had to use for MuJoCo 3.1.1 — this may be genuinely cleaner); (b) generate a static taxel-site XML once and load that. Bumping the submodule is not free — 13 commits including flex work and a timestep change.

**Also worth noting:** `robot_touch_sensor_array_mjx_generated_model.xml` has **429 sites and 421 `<touch>` sensors**, but zero sites named "taxel" — that's the *native touch sensor* model, a different representation with a different naming scheme. Don't confuse the two.

### 1.6 Plan-doc accuracy audit (2026-08-13) — every claim checked against source

`project2_parallel_rl_sim_pipeline.md` was audited claim-by-claim and **patched**. It held up unusually well.

**Verified correct against source:**

| Claim | Verified against |
|---|---|
| 421 native `<touch>` sensors | `grep -c "<touch "` = **421** |
| Touchgrid adds 361 collision geoms | `grep -c "<geom "` = **361** |
| Flex model 385 bodies / 1129 joints | **385 / 1129** (`connect` ≈370, doc says 368 — trivial) |
| 368 taxels | `taxel_layout.N_TAXELS = 368` |
| `nconmax=48`, `njmax=120` | `env_cfg.py:330,332` |
| `sim_dt=0.01`, `decimation=5` → 20 Hz | `env_cfg.py:340-341`; `episode_length_s=50.0` |
| Actor/critic 512-256-128 ELU, 24 steps/env, adaptive KL | `rl_cfg.py:12,22,34,42`; `desired_kl=0.01` |
| Obs set + asymmetric critic terms | `env_cfg.py:72,92-105` — exact match, **no taxels** |
| Multi-GPU via **torchrunx** | `scripts/train.py:209,219` |
| `leapXELA_model`: poetry, `mujoco==3.5.0` | its `pyproject.toml` |
| `leapXelaMjLab`: Python 3.10–3.12, uv | `requires-python = ">=3.10,<3.13"` |
| MJX-JAX has no flex | `mjx/_src/io.py:315` → `NotImplementedError('Flex not implemented for JAX backend.')` |
| Warp supports flex | `mjGEOM_FLEX`, `mjOBJ_FLEX`, `mjEQ_FLEX`, `mjEQ_FLEXSTRAIN`, `nflex` in `_src/types.py` |
| `grasp_touch_test.py` logs 368 taxels → npz + heatmaps | its docstring (also renders an MP4) |
| `VirtualTaxelSensor` algorithm description | read in full — accurate in fine detail |
| `sparsh-skin-sim` CPU, "changed sync in vecenv to async" | commit log, 2026-08-13 |
| `leapXELA/reorient.py`, submodule wiring, 6 branches | all confirmed |

**Corrected in the patch:**
1. **"Every tactile implementation is CPU-only Python"** → wrong (§1.4b). Fixed in §1, §11 interview story, and a new §1.6 added to the plan doc.
2. **"49 commits"** → actually **44**.
3. **`max_contact_points=30` / `max_geom_pairs=12`** → **neither string exists anywhere in `leapXELA_model`.** Flagged as UNVERIFIED in the doc; the surrounding argument still stands. **Find the real values before citing.**
4. **`VERTCOLLIDE`** → not present in `mujoco_warp` under that name (`ELASTICITY` is). Flagged.
5. **§3 hardware** → rewritten with the Spark unified-memory warning.
6. **Added:** Pratik as a second contributor, `contactile_mjlab` prior art, the unpinned `mjlab branch = "main"` (a risk §9 warns about that is *already live*), and the SSH submodule-URL gotcha.

---

## 2. Session log

### 2026-08-13 — session 2 (this one)

**Started with:** a lost session. Memory directory was empty; the previous conversation's history was gone.

**What happened:**
- Established that Claude had no memory of session 1. Reconstructed state from disk.
- Diagnosed the "SSH keeps disconnecting" problem → **not network, host OOM crash**. Full detail in §1.1. This is the session's main finding.
- Read the full plan doc (all 362 lines).
- Surveyed all of Hamid's repos via authenticated `gh` API rather than page scrapes. Verified every §1.5 claim in the plan — they all hold. Findings in §1.3–1.5.
- Discovered the second contributor and what it implies for positioning (§1.4).
- Read `touch_sensor.py` and `taxel_layout.py` in full; recorded the kernel-relevant details (§1.5).
- **Ran a proper redundancy check** (§1.4b) after Afthab pushed back on how thorough the first survey was — the first pass was targeted, not comprehensive. Found `contactile_mjlab` and, more importantly, that MuJoCo Warp already implements native touch sensors on GPU. **The plan's "no tactile on GPU at all" framing needs correcting.**
- **Audited the plan doc claim-by-claim against source and patched it** (§1.6). ~18 specific numbers verified correct; 1 framing error, 1 count error, 2 unverifiable figures flagged, 4 additions.
- Created this log + persistent memory files.

**Nothing was executed on the GPU this session.**

**Decisions:**
- Treat the Spark as the **development, profiling and kernel machine**; the 4096-env points (plan Phase 1 baseline, Phase 6 tactile training) move to a real cluster.
- Re-scope the plan's fixed sweep range N ∈ {16, 64, 256, 1024, 4096} once the Spark's actual ceiling is measured.
- Memory-per-env on unified memory is a **deliverable, not just a nuisance** — plan Phase 3 already asks "does max feasible N drop, by how much?", and the answer is sharper here than on a discrete GPU.

### 2026-08-13 — session 1 (lost, reconstructed)

No transcript survives. From filesystem evidence: the plan doc was written (18:08), a bare `.venv` created (18:35) with mujoco/mujoco_warp/torch, and an attempt made to run ~4096 envs, which crashed the host at ~18:49. No benchmark code was written to disk.

---

## 3. Open questions for Hamid

From plan §10, ordered by how much they'd change the project. The first one could invalidate it, so it goes first.

1. **Is anyone already putting tactile observations into `leapXelaMjLab`?** (Public evidence says no, and Pratik's work is model/dataset-side — but he may have unpushed work.)
2. Which tactile representation is canonical for RL — splat, native `touch` array, or flex/Sparsh-Skin? *Decides the primary kernel target.*
3. Is mjlab/Warp the settled direction, or is MJX-JAX still live?
4. Are `nconmax=48` / `njmax=120` tuned, or placeholders?
5. Where does `sparsh-skin-sim` fit — is tactile RL meant to run there, or in mjlab?
6. Happy for a public repo, and how does he want attribution?
7. **Has he tried native MuJoCo `<touch>` sensors in mjlab?** Warp supports them on GPU already (§1.4b) — if he's already aware, that's the cheap path and it reframes the project around the splat encoder specifically.
8. **What hardware does he train 4096 envs on, and can Afthab get access?** (Now urgent — see §1.1. Also worth telling him about the Spark unified-memory finding; it's directly useful to him.)

## 4. Contribution surface (for "contribute meaningfully")

Ranked by value-to-Hamid per unit effort:

- **GPU tactile observations in `leapXelaMjLab`** — the headline. A new `ObservationTermCfg` in `mdp/observations.py` is a small, clean, reviewable diff. Nobody else is working in that repo.
- **A fast multicore C++/OpenMP encoder** — directly useful to `sparsh-skin-sim`, which is CPU and *currently* fighting vectorised-env throughput (commit `changed sync in vecenv to async`, 2026-08-13). This is a live pain point, today.
- **The cost-of-tactile-sensing measurement** — "what does each sensing mode cost at training scale" is a question he has to answer eventually; nobody has the numbers.
- **The unified-memory / Spark finding** — small but immediately useful if anyone else in the lab has a Spark.
- **A Warp-side contact-force accessor** — whatever we build to replace `mj_contactForce` is reusable infrastructure, not just ours.

## 5. Next actions

1. Write the cgroup-capped memory-ramp harness (N = 16…512, peak RSS + env-steps/sec). **Not yet written.**
2. ~~Clone the repos into `third_party/`~~ — **DONE 2026-08-13.** Both forked to `AfthabShiraz/`, added as submodules pointing at the forks, `upstream` remotes wired, global `insteadOf` rewrite set so the SSH submodule URL resolves.
2b. **Resolve the missing-taxel-sites problem (§1.5b) — this now precedes all encoder work.**
3. Phase 0 proper: CPU tactile reproduction (`reading_contact.py`, `grasp_touch_test.py`), oracle fixtures, `HYPOTHESES.md` written and dated **before** measuring.
4. Email Hamid — §3, leading with Q1 and Q7.
5. Re-scope §4's sweep range once (1) gives a number.
