# Project Log — Tactile Observations at Training Scale

**Purpose:** durable running record of what we've done, decided, and learned, so no context is lost between sessions. Claude updates this at intervals during work. Newest session at the top of §2.

**Companion docs:** `project2_parallel_rl_sim_pipeline.md` is the *plan* (stable). This is the *state* (changes constantly). If the two disagree, this file is what actually happened.

**How to work with Afthab (set 2026-08-13):** he is learning MuJoCo *while* building — understanding the system is an explicit goal, not a by-product. **Explain in short, friendly chunks and stop for questions.** No walls of text, no stacked tables, no six-section replies. One idea per turn, then pause and offer to go deeper. Introduce MuJoCo concepts when the work reaches them rather than as upfront theory. Precision and honesty stay the same; only the volume per turn changes.

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

> **CORRECTION, same day, before acting on any of this.** The first pass checked `leapXela_generated_mjx.xml` and `..._CoACD.xml` but **not `..._Box.xml`**, which is the variant `get_hand_xml()` defaults to. Corrected counts of `<site name="taxel_NNN">`:
>
> | file | pinned `4e8003f` (what mjlab loads) | current `main` |
> |---|---|---|
> | `leapXela_generated_mjx.xml` | 0 | 0 |
> | `leapXela_generated_mjx_Box.xml` | **0** | **368** |
> | `leapXela_generated_mjx_CoACD.xml` | 0 | 0 |
> | `leapXela_generated_mjx_flex_sensorCoACD.xml` | — | 368 |
>
> **The section's conclusion still holds for what mjlab actually trains on today** — every variant at the pinned June-11 commit has zero taxel sites. But it is wrong about current `main`, and the two facts below change the plan.

**(a) The 368 taxels on current `main` are not what the splat encoder needs.** They sit on **368 individual flex bodies** — `flex_if_tip_1..30`, `flex_uspa46_1_0..23`, etc., one taxel site per body — children of the rigid links. That is the **flex/bubble skin representation** (plan §2.2c, hypothesis H4). The splat encoder needs 368 sites on the **12 rigid links**, because `VirtualTaxelSensor` groups taxels by body and Gaussian-splats each contact across *that body's* taxels. On the flex model every body owns exactly one taxel, so the splat degenerates to a single taxel per contact — the algorithm silently stops being itself. **The two representations need different models. `BODY_MAP` is still required for the splat path.**

**(b) ⚠ Bumping the submodule silently swaps the robot.** Same filename, completely different model:

| `leapXela_generated_mjx_Box.xml` | bodies | joints | connects | taxel sites |
|---|---|---|---|---|
| pinned `4e8003f` (Jun 11) | **17** | 25 | 0 | 0 |
| current `main` (Aug 12) | **385** | **1129** | **368** | 368 |

A submodule bump would hand mjlab a 385-body, 1129-joint, 368-constraint deformable hand in place of a 17-body rigid one, under an unchanged path. At 4096 envs that is not a small change — it is hypothesis H4 being triggered by accident, with no measurement and no decision. **Do not bump the submodule casually; pin deliberately and measure the flex model as its own representation.**

**(c) `BODY_MAP` was re-validated against the pinned model** (`third_party/leapXelaMjLab/.../assets/leapXELA_model`, both `_Box` and plain): identical result, 8/12 frames agree, same four fingertips off by the same 10.33 mm. The mapping work is unaffected by this correction.

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

**RESOLVED 2026-08-13 — see `explore/02_map_bodies.py`.** Both lineages are structurally identical (palm + 4 chains × 4 links), so the map is derivable from the kinematic tree, disambiguated by the layout's own `PATCH_TO_HARDWARE` names.

**⚠ Chain order is reversed from what the names suggest: `finger` is the RING finger, not the index.** Mapping by name order — the obvious guess — puts every index taxel on the ring finger and every ring taxel on the index. Plausible numbers, silently wrong. The map is:

```python
BODY_MAP = {
    "leap_hand_xela_back_cover": "palm",
    "p3_unified": "rf_px",   "p2_unified": "rf_md",   "finger": "rf_ds",     # RING
    "p3_unified_2": "mf_px", "p2_unified_2": "mf_md", "finger_2": "mf_ds",   # MIDDLE
    "p3_unified_3": "if_px", "p2_unified_3": "if_md", "finger_3": "if_ds",   # INDEX
    "thp1_unified": "th_px", "thumb": "th_ds",                               # THUMB
}
```

**Frame agreement, measured by body-local mesh bounding box** (*not* centroid — corresponding meshes are re-exports whose vertices differ by up to 1.5 mm, which moves a centroid but not a bounding box; this cost one wrong intermediate result. ⚠ **The original reason given here — "the RL meshes are decimated" — is wrong; see §1.5b-bis.** The choice of bbox over centroid was still correct, for the reason now stated):

| Bodies | Taxels | Result |
|---|---|---|
| palm + all 8 px/md links | **232** | **bbox identical to 0.000 mm — taxels transfer unchanged** |
| `if_ds`, `mf_ds`, `rf_ds`, `th_ds` | **136** | **frame origin agrees, but the RL fingertip is a physically different, larger part** |

Fingertip detail (`finger_3`→`if_ds`): faces at `min_x`, `max_y`, `min_z` align **exactly**, so the mounting frame is shared; the RL tip extends **+10.33 mm** in −y, +4.85 mm in +x, +4.6 mm in +z. Tactile tip `30 × 61.98 × 34.6 mm` vs RL tip `34.85 × 72.31 × 39.2 mm`. Identical across **all three** RL variants (`Box`, plain, `CoACD`) — so this is not a fingertip-variant choice, the two lineages genuinely carry different tip designs.

**Consequence:** 232 of 368 taxels can be injected today with no transform. The 136 fingertip taxels would sit *inside* the RL tip rather than on its surface, so they need either a matching tip model or re-projection onto the RL tip surface. **This is now a precise question for Hamid** rather than an open-ended one: *"the tactile lineage's fingertip is 10 mm shorter than the mjx one — which is the real robot, and is there a tip variant that matches the XELA-sensorised tip?"*

**Original recommendation, retained for context:** (i) ask Hamid — he built both lineages and may already know they correspond, or have the transform; (ii) derive the name map and **validate geometrically** — render the injected sites and confirm they sit on the skin, and check each site's distance to the nearest mesh surface numerically; (iii) only then port the injection. Do **not** skip (ii).

Rejected alternatives: making mjlab load Lineage B (large change; `robot.xml` needs the compatibility fixups `scene_builder.py` applies), and regenerating Lineage A with taxels (needs taxel positions in Lineage A geometry — same unsolved problem).

**This is a strong candidate for the first real contribution:** it blocks tactile-on-GPU entirely, it's tractable, and the artifact — a validated body map plus taxel sites in the mjx model — is reusable by Hamid, Pratik and `sparsh-skin-sim` alike.

### 1.5b-bis — the two lineages share a CAD ancestor, and only the fingertip was revised (2026-08-15)

Established by hashing the mesh assets directly, which is stronger evidence than
the compiled-model bounding boxes above and **corrects two claims in §1.5b**.

Both lineages ship an `assets/` directory; **17 `.stl` filenames appear in both**:

| | count |
|---|---|
| shared `.stl` files | 17 |
| **byte-for-byte identical** | **10** |
| same triangle count | **17 of 17** |
| different triangle count | **0** |

The byte-identical ten are `leap_hand_xela_back_cover`, `leap_hand_xela_bottom`,
`leap_hand_xela_motorholder`, `leap_hand_xela_side_cover`, `outer_skin_v1_3_1`,
`palm_face`, `top_palm`, `thconnector`, `clipper`, `uspa46`.

**Correction 1 — "two independent CAD exports have no reason to agree" is wrong.**
Independent exports do not produce byte-identical files. These hands come from one
CAD project exported twice. The mapping risk §1.5b worried about was real to check
but the underlying models are far more closely related than assumed.

**Correction 2 — the RL meshes are NOT decimated.** All 17 shared meshes have
*identical* triangle counts (e.g. `p2_unified.stl`: 17,076 in both). The differing
files are re-exports: same triangle count, same bounding box to 0.01 mm, vertices
differing by ≤1.5 mm. Using bbox over centroid was still the right call — 1.5 mm of
vertex drift moves a centroid — but the reason recorded in §1.5b was not the real one.

**The fingertip is a genuinely different part, and this is the useful finding.**
It is the one component that does not share a filename at all:

| lineage | mesh file | dimensions (mm) |
|---|---|---|
| RL / mjx | `fingertop_unfied.stl` | 39.20 × 34.85 × **72.31** |
| tactile | `fingertip_fix.stl` | 34.60 × 30.00 × **61.98** |

72.31 − 61.98 = **10.33 mm**, reproducing the figure measured from the compiled
models exactly, now traced to source geometry. Mesh-to-body confirmed in both XMLs:
tactile `finger` uses `fingertip_fix`; RL `if_ds`/`mf_ds`/`rf_ds` use
`fingertop_unfied`, and `th_ds` uses `thfingertip_unified`.

**So this is not an irreconcilable mismatch between two modelling efforts — it is
one part that was revised, captured at two different revisions.** There is a
definite answer to which one is the physical robot, and Hamid can give it in a
line. **Ask it in this form**, not the §1.5b form:

> Your mjx and tactile models share CAD ancestry — ten of the mesh files are
> byte-identical. But the fingertip is a different part between them:
> `fingertop_unfied.stl` is 72.31 mm along its long axis where `fingertip_fix.stl`
> is 61.98 mm. The taxel layout was authored against the shorter one. Which is the
> physical robot?

Incidentally the unshared meshes show what each lineage was tooled for: RL carries
`outer_skin_collision_1..6.stl` and `sphere.stl` (simplified collision primitives);
tactile carries `pointcloud.stl`, `4x6.stl`, `4_6_origin.stl` (taxel-layout
scaffolding). Same robot, two jobs.

**Consequence:** before any encoder work, the 368 taxel sites must exist in the model mjlab loads. That is model plumbing, not kernel work, and it is Phase 0/1. Options to weigh: (a) bump the submodule to current `main` and port `build_scene_xml`'s site injection into the mjlab spec pipeline (mjlab uses `MjSpec`, which is *newer and nicer* than the ElementTree hack `scene_builder.py` had to use for MuJoCo 3.1.1 — this may be genuinely cleaner); (b) generate a static taxel-site XML once and load that. Bumping the submodule is not free — 13 commits including flex work and a timestep change.

**Also worth noting:** `robot_touch_sensor_array_mjx_generated_model.xml` has **429 sites and 421 `<touch>` sensors**, but zero sites named "taxel" — that's the *native touch sensor* model, a different representation with a different naming scheme. Don't confuse the two.

### 1.5c Taxel injection into the pinned rigid model — works (`explore/03_inject_taxels.py`)

368 sites injected via **`MjSpec`** into `third_party/leapXelaMjLab/.../leapXela_generated_mjx_Box.xml` (pinned `4e8003f`, 17 rigid bodies, 0 pre-existing taxels). Compiles; model goes 67 → 435 sites. Because mjlab builds its hand through `MjSpec` too (`robots/leap_xela.py:get_spec`), this transfers into an mjlab `spec_fn` nearly verbatim — no ElementTree surgery.

Rendering confirms an anatomically sensible map: three 4×6 palm patches, 4×4 pads on the proximal/middle links of each finger, and dome-shaped fingertip clusters. **The sites are only visible with geoms hidden** — see below.

**Burial is normal, and it is not the correctness criterion.** All 368 taxels sit *inside* their link mesh in the mjlab model — but so do **248/368 in the reference lineage-B model**, which is the one the validated CPU encoder actually uses. Chasing "taxels should be on the surface" would have been a wild goose chase.

The criterion that *does* matter is the encoder's patch-locality gate:
```python
weights[np.abs(local[:, 2]) > self._kernel_cutoff] = 0.0
```
`local_z` is the contact's offset along the taxel normal, so a taxel buried *d* below the surface sees `local_z ≈ d` for a contact directly above it. **If `d ≥ kernel_cutoff` that taxel can never fire.** Defaults (`visualize_taxel_layout.py`): `kernel_sigma=0.0035`, `kernel_cutoff=0.01` (10 mm). Note the Gaussian weight uses only the *in-plane* distance — burial does not attenuate it, it only gates.

Measured burial by ray-cast along each taxel's own +z, **restricted to its own body** (an early version traced all geoms and struck neighbouring fingers in the rest pose — the numbers were meaningless until isolated):

| | mjlab (lineage A) | reference (lineage B) |
|---|---|---|
| palm | 3.00 mm | 3.00 mm — identical |
| px links | 5.00 mm | 9.76 mm |
| md links | 4.50 mm | 9.76 mm |
| fingertips | 2.17 mm | 0.00 (not buried) |
| **all 12 bodies vs 10 mm cutoff** | **max 5.00 mm — OK** | 9.76 mm — OK but within 0.24 mm of the gate |

**Conclusion: every body clears the cutoff, so the splat's locality gate behaves the same in mjlab as in the reference.** The mjlab model is in fact the *safer* of the two — the reference's 9.76 mm sits 0.24 mm under its own cutoff, so `kernel_cutoff` is load-bearing and geometry-dependent there. Worth stating in the write-up, and worth not changing casually.

**Still open:** the four fingertips remain a geometrically different part (~10 mm longer, §1.5b), so their 136 taxels are placed on a tip that is not the one they were calibrated against. Burial there is small (2.17–2.85 mm) and within the gate, but the *lateral* placement is unverified. This is the question for Hamid.

### 1.5d The reference encoder runs on the mjlab hand — oracle banked (`explore/04_run_encoder.py`)

`VirtualTaxelSensor` now produces real taxel readings on the model mjlab trains on. First working touch on the RL hand. Busiest frame: 28 taxels active, peak 5.06 N, channels behaving sanely (`normal_z` positive under compression).

**⚠ Porting gotcha — injecting the sites is not sufficient.** `VirtualTaxelSensor.__init__` groups taxels by body:
```python
body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, entry.body)
```
`entry.body` holds *tactile-lineage* names, absent from the RL model, so every lookup returns `-1`, all 368 taxels collapse into one bogus group, and no contact ever matches. **The encoder runs without error and outputs all zeros.** Fixed by `taxel_map.remap_layout()`, which rewrites `entry.body` through `BODY_MAP` — deliberately remapping the *layout* rather than patching the encoder, since the encoder is the correctness oracle and must stay byte-for-byte identical. Any GPU port needs the same remap.

**Oracle fixture:** `explore/out/oracle_fixture.npz`, 30 frames sampled through a scripted grasp, including `ncon=0` cases. Each frame stores the encoder's *inputs* (contact pos, frame, geom ids, 6-vector force, plus `site_xpos`/`site_xmat`/`qpos`) alongside its 368×3 output, so any later implementation can be validated **without re-running MuJoCo** — which also sidesteps the Warp-vs-CPU contact discrepancy problem (plan §7) for pure encoder validation.

**Caveat on coverage:** in this grasp only the palm fired. Contacts do occur on the finger links (`if_md_uspa44`, `if_tip` — see `01_see_contacts.py`), but the 4×4 pads cover one face of each link, so a contact on an uncovered face correctly yields zero weight and is skipped. The fixture therefore exercises the palm path well and the finger path barely. **Before trusting a kernel against it, extend the fixture with a grasp that loads the fingertips.**

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

### 2026-08-14 — session 6: every sweep re-run under real grasp motion, and the report written

**Why.** Every throughput number banked before this session was taken at one
frozen pose: 16 actuators to 60% of range, settle, hold for the whole timed
region. ~7.5 live contacts/env, constant, and *identical in every world*. That
is a weak test of a kernel whose cost tracks live contacts, and it flatters the
Triton path's ratio against a dense baseline that is contact-count blind. Both
numbers were honest; the operating point was soft.

**`GraspDriver`, new in `harness.py`, behind `--motion {static,grasp}`.** Drives
the seven patterns from `explore/05_grasp_motions.py` (themselves a verbatim
port of his `sparsh-skin-sim/util/motion_util.py`). World `w` runs pattern
`w % 7` at a phase offset, so the batch spans the whole modulation at any
instant. `ctrl` is rewritten each step from a precomputed GPU table — one gather,
one copy, no host round trip — and charged to every variant equally, `none`
included. `--motion static` reproduces the old operating point exactly, so the
banked CSVs stay reproducible.

**Contact regime it produces at N=4096:** per-world count 0 to 25, population
mean 7.33/env against the frozen pose's 7.5. So the batch now contains idle
worlds and loaded worlds simultaneously, which is what puts real load imbalance
in front of a one-block-per-(env, taxel-tile) kernel.

**The result survives, and improves:**

| path | frozen steps/s | grasp steps/s | frozen cost | grasp cost |
|---|---|---|---|---|
| physics only | 128,061 | 126,637 | 1.00× | 1.00× |
| eager splat | 28,006 | 27,642 | 4.57× | **4.58×** |
| Triton splat | 116,961 | 125,535 | 1.09× | **1.01×** |
| native touch | 125,877 | 117,601 | 1.02× | **1.08×** |
| touchgrid `object` | 29,440 | 27,942 | 4.35× | **4.53×** |

Encoder stage 2.394 ms under motion against 2.432 ms frozen; eager unchanged at
112.99 vs 112.19 ms, exactly as its contact-blind `B × C × T` shape predicts.

**Why Triton *improves* rather than degrades — mechanism, not a lucky draw.**
Pass 2 is indexed by (env, taxel tile): a program walks its own env's contact
slots, finds them invalid, and leaves the loop. An idle world costs a scan of its
slot array, not an evaluation, and with no atomics and no cross-world sync it
does not hold up a loaded world either. Under motion many worlds sit below the
frozen pose's uniform occupancy, so total work falls.

**⚠ Read 1.01 as "not resolvable", not as a bound.** At N=1024 the Triton path
measures 76,891 against a physics-only 72,804 — nominally faster than doing no
tactile work at all. The honest statement is that under motion at large N the
cost is not distinguishable from run-to-run variation. And the upper end is still
unmeasured: no condition holds every world near the 48-slot cap, and the
synthetic fully-occupied case costs 1.27 ms (2.2× the measured regime).

**Touchgrid under motion sharpens the budget finding.** Peak per-world contact
count rises to **390**, against the 232 measured at the frozen closing grasp, with
no overflow at the 256-slot allocation. 151,669 patch-cube contacts and **zero**
patch-patch, confirming the H3 mechanism under motion too. **The budget a
geom-based grid needs is a property of the motion, not of the pose you benchmark
— sizing `nconmax` from a static grasp under-provisions it.**

**A verification gate that cried wolf.** The first grasp-motion touch sweep
aborted at N=1: the gate read all 421 sensors at a single instant, got zero, and
stopped. That was a false alarm — the `tap` pattern deliberately releases toward
pregrip at 1.5 Hz, and at N=1 there is one world running one pattern at one
phase, so a quiet frame means an open hand, not a broken readout. The gate now
samples a window (≤200 steps) and takes the best frame; it still catches a sensor
array that *never* fires, which is what it is for. Sweep then completed 13/13.

**Also this session:** `report.tex` written — the full IEEE write-up covering all
four representations, the profiling verdict, the kernel, the mjlab integration
and the threats to validity.

**⚠ Correction to a report claim, made 2026-08-15.** The report stated that the
grasp-motion native-touch sweep never completed and that
`scale_sweep_grasp_touch_dt01.csv` recorded a failed run. That was written at
19:39 against a memory of the aborted first attempt; the gate had been fixed and
the sweep re-run successfully at 16:14 the same day, 13/13 rungs. Corrected in
`report.tex` (Results, Threats, Future Work, appendix). **Lesson: when a run is
re-done after a fix, update the prose in the same sitting — a stale "this
failed" is worse than no claim, because it is checkable and wrong.**

### 2026-08-15 — session 7: repo audit, four fixes

Read the repo end to end and reconciled the documents against the banked data.
Four things were out of step; all fixed.

1. **The session-6 entry above did not exist.** The grasp-motion work was in the
   code, the CSVs and the report, and nowhere in this log or in `HYPOTHESES.md`.
2. **`report.tex` claimed the grasp-motion touch sweep failed** — see the
   correction above.
3. **`SplatEncoder.C` still ignored `--nconmax-per-env`** (flagged as a latent
   issue in session 4 and still live). `harness.py` built the encoder without
   `contacts_per_env`, so `--tactile splat --nconmax-per-env 256` allocated 256
   slots per world in `put_data` while the encoder bucketed into 48, silently
   discarding every contact past the 48th and reporting it as a *drop* rather
   than as a misconfiguration. Fixed by threading `nconmax_pe` through.
   Verified: at N=16 the default path is unchanged (`contacts_per_env_cap` 48,
   `naconmax_total` 768, check A 2.98e-07 PASS) and `--nconmax-per-env 256` now
   reports cap 256 / total 4096, check A 4.77e-07 PASS.
4. **`README.md` had drifted a long way** — it was frozen at the post-kernel
   state of session 3. Fixed: headline and every results table re-based on the
   `_dt01` sweeps (4.57× → 1.09×, Amdahl 4.34×, crossover share 53.3%); status
   corrected (H3/H4 resolved, only H5 open); added §4.9 (the other three
   representations plus the `nconmax` law), §4.10 (the mjlab integration), the
   grasp-motion result, and the widened-oracle findings F1–F3 in §5; sweep
   commands updated for the new flags.
5. **`README.md` mis-attributed `VirtualTaxelSensor` to Hamid.** It is **Pratik
   Ingle's** work (`5f305a8`, 2026-08-03). `report.tex` had already been
   corrected; the README had not. It is the oracle for every number in this
   project, so this is the attribution that most needs to be right.
6. `analysis/plots.py`'s docstring listed the pre-`_dt01` CSV names while its
   code defaults to `SWEEP_SUFFIX="_dt01"`. Docstring corrected.

**Verification run this session:** all **11 tests pass** (v1 + v2 oracle, Triton
vs both fixtures, edge cases, determinism 20/20 bit-identical at B=1024),
float64 semantic bound 4.16e-07 on the 632-frame fixture.

**Flex ratio mis-attributed in the report's Table I — fixed.** The footnote said
the ≈540× ratio was "per-step latency at N=1024". Two errors: **there is no N=1024
flex latency measurement** (the rungs are N = 1, 16, 64, 256; only a *memory*
figure exists at 1024), and raw per-step latency at N=256 gives **107.6×**, not
540 — the 540 needs the ×5 matched-simulated-time correction, so as written it
looked like a 5× error. Both derivations agree once stated properly (538.1 via
latency×5, 539.6 via equivalent throughput). Footnote rewritten in `report.tex`
and `README.md`; the abstract, flex section and conclusion were already correct.
**Found by Afthab asking what the sentence meant.**

**Mesh-asset provenance checked (§1.5b-bis), prompted by Afthab asking whether the
two hands come from the same CAD model.** They do — 10 shared `.stl` files are
byte-identical. This corrects two claims in §1.5b (the "independent exports"
framing and the "RL meshes are decimated" reason) and, more usefully, localises
D4 to a single revised fingertip part with a definite answer. Recorded as F4 in
`HYPOTHESES.md`; report's method and threats sections updated. **Worth noting how
this surfaced: it came from a question during a read-through, not from the
measurement plan.**

**Figures audited, and two added.** Checked whether the report's 8 figures had
gone stale: regenerated them and all 8 came back **byte-identical**, and every
number `plots.py` derives matches the report. They were current; the gap was
*coverage* — every figure was about physics vs eager vs Triton, and nothing from
sessions 4–6 had one.

* **`benchmarks/contact_budget_sweep.py`, new.** The `nconmax` law was H3's most
  generalisable result and it lived only as one-off numbers in the register. Now
  a proper sweep: one memory-capped child per budget, a CSV, a least-squares fit
  printed at the end. Run **twice**, on scenes 5× apart in geom count, because
  the claim is that the geometry is irrelevant:

  ```
  bare scene (71 geoms)         ms/step = 5.51 + 0.1428 x nconmax   R² = 0.9978
  grid, collide off (371 geoms) ms/step = 4.73 + 0.1436 x nconmax   R² = 0.9955
  ```

  **Slopes agree to 0.6%**, occupancy pinned at 7.4–7.7 contacts/env at all
  twelve points, throughput 78,045 → 17,062 env-steps/s (4.6×). The original
  0.145 slope **reproduces to within 1%** — worth knowing, since it was banked
  from single runs. Banked as `contact_budget_none_n1024.csv` and
  `contact_budget_grid_off_n1024.csv`.
* **Figure I** — the comparable representations at N=4096 as one bar chart.
  **Flex is deliberately omitted rather than plotted**, because H4 says in terms
  that its `env_steps_per_sec` is not a like-for-like bar; it stays in the
  report's table where the dagger travels with it. The touchgrid's bar is
  annotated with the larger budget it needs.
* **Figure J** — the budget law, both series with fitted lines. The two lines
  sit almost exactly on top of each other, which is the whole argument.
* Report updated: both figures wired in with captions, §H3's scan paragraph
  rewritten around the banked data, the Future Work item narrowed (the sweep now
  exists; what is still open is whether the slope varies with N or with the
  model), and the appendix gained the commands and the two CSVs.

### 2026-08-14 — session 4: the other three tactile representations measured

**Goal:** run the experiments for the three representations that had not been
swept. The splat was already done (torch / compile / triton, 13 rungs). This
session did native `<touch>`, the binary touchgrid, and flex.

**Where each stood at the start:** the `--tactile touch` path was already written
into `harness.py` (uncommitted, 04:15 that morning) but had **never been run** —
no CSV existed. Touchgrid and flex did not exist at all.

#### Native `<touch>` (421 sensors) — swept, and it is essentially free

13/13 rungs to N=4096, `benchmarks/results/scale_sweep_touch_dt01.csv`.
**125,877 env-steps/s at 4096 against bare physics's 128,061 — a 1.7% tax.**
The `naconmax × 421` dispatch measures 0.005–0.14 ms/step, at or below the
timing noise floor at every single N. It never resolves above noise.

Two things made this measurable at all, both already documented in the harness:
the sensors had to be *injected* into the RL scene (the supervisor's standalone
touch model reads exactly zero — its sensor pads were reduced to visual meshes,
so the nearest contact is 9.42 mm from a 1.00 mm sensor zone), and the cost had
to be extracted as a difference of fused loops, because Warp computes touch
inside `mjw.step` and there is no stage to put a clock around.

**Tell Hamid this one.** It is the cheap path, it already works on GPU, and it
costs ~2% at training scale.

#### Binary touchgrid (300 patch geoms) — the session's real finding

Full write-up in `HYPOTHESES.md` H3. The short version, because the mechanism is
not what anyone predicted:

* At the supervisor's `nconmax=48 / njmax=120` the touchgrid **crashes**
  mujoco_warp — illegal memory access in `ccd_kernel`, not a graceful drop.
  Measured requirement is ncon 232 / nefc 944 vs the bare scene's 24 / 112.
* The contact explosion is **intrinsic**: 223 of the 232 peak contacts are
  patch-vs-cube, with zero self-collision.
* But the contacts are **not what costs**. Holding the budget fixed, turning the
  patches' collisions on costs nothing (29,440 vs 28,926 env-steps/s at N=4096).
  300 extra non-colliding geoms cost 2.4%.
* **The cost is the allocation.** `ms/step ≈ 5.3 + 0.145 × nconmax` at N=1024
  with the live contact count pinned at 7.4/env. Raising `nconmax` 48→256 costs
  71% of throughput with identical physics. Raising `njmax` costs nothing.
* Mechanism confirmed in source: mujoco_warp sizes its launches off
  `d.naconmax`, the allocation, not the live count — `collision_convex.py:1250`,
  `constraint.py:5327`, `smooth.py:2941` (`dim=(nacttrnbody, naconmax, nv)`),
  and others.

**Consequence worth carrying into every later result: `nconmax` is a throughput
parameter, not a safety margin.** This reframes open question 4 for Hamid from
housekeeping into something with a 3.4× number attached.

**A wrong turn, recorded because the correction matters.** The first reading of
the 232 contacts blamed D4 (patches injected into the RL lineage's
differently-shaped links), on the evidence that the donor model alone peaks at
38 contacts. That control was invalid — **the donor model standing alone has no
cube in it**. Classifying contacts by what they touch is what settled it. The
misleading comment in `build_touchgrid_model` was corrected in the same session.

#### Flex / bubble skin (H4) — it runs on Warp, and it is ~540× slower

`benchmarks/flex_probe.py`, deliberately separate from `harness.py`: the flex
hand is a different body tree (387 bodies / 1120 joints / 18 flexes / 386 eq
constraints) in the *tactile* lineage, not something injectable into the pinned
rigid RL model. Its `env_steps_per_sec` is therefore **not** a like-for-like
fourth bar in the cost-of-tactile-sensing figure; `ms_per_step` is a legitimate
order-of-magnitude comparison and that is what gets reported.

* `put_model` succeeds — **flex is viable on MuJoCo Warp**, as the plan
  suspected and MJX-JAX cannot do at all.
* `put_data` demands `njmax ≥ 3030` per world, and says so *cleanly* at
  allocation time. Worth contrasting with the touchgrid, which announces the
  same class of problem by dying inside a CUDA kernel.
* **⚠ The first flex numbers in this log were wrong and have been corrected.**
  `flex_probe.py` forced `timestep=0.01` to match `env_cfg.py`. The flex model
  declares **`timestep=0.002` with 50 solver iterations** and does not survive
  0.01 — MuJoCo prints `Nan, Inf or huge value in QACC ... the simulation is
  unstable` and the Kelvin-Voigt readout returns ~1e5 N. Everything measured at
  dt=0.01 is superseded. A second, smaller bug rode with it: the settle was a
  fixed 60 steps, which is 0.6 s at dt=0.01 but 0.12 s at 0.002 — too short for
  the hand to close, so the first corrected run reported zero contacts. Settle is
  now specified in simulated seconds. Found while debugging the splat-vs-flex
  comparison, which is what surfaced the instability.
* Corrected, at native settings, gripping, and rescaled to matched simulated time
  (a 0.002 s step advances 1/5 as far, so raw steps/s overstates flex 5x):
  N=1 → 137.1 ms/step (**1** equiv env-step/s); N=64 → 256.9 ms (**50**, vs
  rigid 15,999 → **~320×**); N=256 → 719.2 ms (**71**, vs rigid 38,310 →
  **~540×**).
* Solver iterations are NOT the driver (50 vs 5 → 159.8 vs 159.1 ms). The
  timestep is.
* **Where the time goes** (N=64 ablation): ~37% constraint solver, ~63% forward
  dynamics over 387 bodies / 1120 DoF, **~1% contacts**. A kernel cannot help —
  it is all inside `mjw.step`, and the readout we would control is a few thousand
  flops against ~200 ms.
* Memory never binds: 0.75-0.81 GB throughout. This is the second time D7's
  memory-ceiling assumption has failed to bind; time is the constraint.

#### Harness changes (all in `benchmarks/`)

* `--tactile touchgrid` with `--touchgrid-collide {object,naive,off}`;
  `build_touchgrid_model`, `TouchgridReadout`, `verify_touchgrid`.
* `--nconmax-per-env` / `--njmax-per-env` are now flags rather than constants,
  which is what made the budget isolation above possible. Defaults are unchanged
  at the supervisor's 48 / 120, so every banked sweep stays reproducible.
* `benchmarks/flex_probe.py`, new.
* **Latent issue, not fixed:** `SplatEncoder.C` still reads the module constant
  `NCONMAX_PER_ENV` rather than the CLI value, so running `--tactile splat` with
  a raised budget would silently mismatch. Harmless today (the splat sweeps all
  used defaults) but it will bite whoever tries splat-at-raised-budget.

#### Reproduction check — because two GPU jobs briefly overlapped

A short flex `put_model` probe was running on the GPU during the tail of the
touchgrid sweeps. Contended timings are worthless, so the key rungs were re-run
on an idle GPU:

| measurement | banked | re-run idle |
|---|---|---|
| physics only, N=4096 | 128,061 | 128,991 |
| physics only, N=1024 | 78,349 | 74,201 |
| touchgrid `off`, N=4096 | 124,963 | 127,291 |
| touchgrid `object` bigbudget, N=4096 | 29,440 | 29,902 |

All within ~5%, so the overlap did not distort the sweeps. The `nconmax`
isolation and scan — the runs the H3 mechanism claim actually rests on — were
made with the GPU idle throughout.

**Practice note:** run one GPU job at a time on this box, and re-run rather than
reason about it when that slips. A separate contaminated measurement in this
session (a `--tactile none` regression check taken while flex N=1024 was
running) read 4× slow and would have looked like a real regression.

**New CSVs:** `scale_sweep_touch_dt01.csv`,
`scale_sweep_touchgrid_object_dt01.csv`,
`scale_sweep_touchgrid_off_dt01.csv`,
`scale_sweep_touchgrid_off_bigbudget_dt01.csv`.

**Not done this session:** figures were not regenerated for the three new
representations, and `report.tex` was not updated. `HYPOTHESES.md` H3 is
resolved; H4 has the numbers but its resolution entry still needs writing.

### 2026-08-14 — session 5: into the supervisor's environment, and a comparison that isn't valid yet

#### Tactile observations now run inside `leapXelaMjLab` (rungs 0–3 of 4)

**`uv sync` works on the Spark.** mjlab 1.5.3 pinned at `f643d24` by his own
`uv.lock` (so reproducible, and the same version he gets), torch 2.13.0,
warp-lang 1.15.0, mujoco/mujoco_warp **3.10.0.3** — note that is *older* than the
3.11.0 our benchmarks used. Installed into `third_party/leapXelaMjLab/.venv`;
our own `.venv` is untouched.

**Working, end to end.** The actor observation goes from 57 to **1161** dims,
1104 of them live 3-channel taxel readings from the Triton kernel, computed every
control step inside his task with his rewards, resets and randomisation.

**Cost in the real environment, N=2048:**

| | mjlab as shipped | with the `site_pos_w` fix below |
|---|---|---|
| no tactile | 16.16 ms/env-step | 14.05 ms |
| **368 taxels** | 50.70 ms (**3.14×**) | **16.67 ms (1.19×)** |

**1.19× is the real answer** and it corroborates the harness's 1.17× prediction.
Read the diagonal: with the fix applied, full tactile costs about what his
current no-tactile setup costs today.

**Three portability traps, all of which fail silently or fatally:**
1. **mjlab namespaces everything** — bodies are `robot/palm`, the cube geom is
   `cube/cube`, sites become `robot/taxel_017`. `BODY_MAP` holds bare names, so
   without a prefix every lookup misses. `prepare_static` raises rather than
   zeroing, so it fails loudly — but only because the prefix is threaded through
   (`SplatEncoder(name_prefix=...)`, new).
2. **`TaxelEntry` is frozen** — the prefix rewrite must use `dataclasses.replace`,
   as `remap_layout` already does.
3. **`wp.set_stream` segfaults inside mjlab.** `Simulation.__init__` captures
   `step`/`forward`/`reset`/`sense` into CUDA graphs bound to the Warp stream
   current at capture time (`sim.py:347`); observation terms are built *after*.
   Repointing the stream under a captured graph dumps core. The harness does this
   legitimately because it owns its Warp context. Inside mjlab, order with
   `torch.cuda.current_stream().wait_stream(...)` instead — measured at 0.02 ms.

New: `src/mjlab_tactile/taxel_term.py`, `benchmarks/mjlab_rung3.py`,
`benchmarks/mjlab_diag.py`, `benchmarks/mjlab_verify_patch.py`.

#### An mjlab bug worth 3.31×, unrelated to tactile

`EntityData.site_pos_w` returns `self.site_pose_w[..., 0:3]`, and `site_pose_w`
gathers every site's position **and** 3×3 orientation, runs `quat_from_matrix`
over all of them, builds an `(N, nsite, 7)` tensor — then three columns are
sliced out and the quaternions discarded. The reorient task does this twice per
step to fetch **one** site's position (`_palm_pos_w`, actor) and four
(`fingertip_positions_rel_palm`, critic).

With the stock 68 sites that is wasteful but cheap. With 436 it becomes the
largest cost in the environment. Patching that one property: **49.92 → 15.07 ms**
at N=2048. It also speeds up the *no-tactile* env 16.16 → 14.05 ms, which is the
proof it is not our bug.

**Verified not to change results** (`mjlab_verify_patch.py`) — this was checked
only after Afthab challenged a speedup reported without verification, which was
the right call:
* `torch.equal(site_pos_w, site_xpos[:, ids])` → **True**, max|diff| **0.0** over
  64×435×3. `cat([pos, quat])[..., 0:3]` is `pos` by construction.
* step-0 observations bitwise identical.
* no in-place writes to any of the six `*_pos_w`/`*_quat_w` properties anywhere
  in mjlab or our code — so returning a view rather than a fresh tensor is safe
  today, though a PR should say so.
* Multi-step divergence is **not** evidence either way: the env is not
  reproducible run-to-run (domain randomisation), and unpatched-vs-unpatched
  diverges as much as unpatched-vs-patched.

**Upstream state:** the pattern is in **six** property pairs, still present on
`main` today. `site_*` and `geom_*` are the expensive ones (they call
`quat_from_matrix`). mjlab is 2.8k stars, Apache-2.0, and merges outside PRs
(18 of the last 60 from 8 non-maintainers). **No PR opened yet** — Afthab's call.

#### Splat vs flex — the experiment is built, and is NOT valid yet

Motivation: his dataset pipeline records **flex** quantities, and flex cannot run
at RL scale, so if flex-like signal is ever wanted in a policy something fast must
stand in for it. Is the splat encoder that thing?

Built: `explore/08_splat_vs_flex.py`, `explore/09_render_splat_vs_flex.py`,
`explore/vendor/flex_util.py` (his Kelvin–Voigt estimator, vendored unmodified
with provenance).

**Solved:** patch correspondence — 18/18 paired by rest-pose geometry,
size-constrained, one-to-one, residuals 2.5–10.4 mm. Deliberately NOT by name:
the splat patch ids (`4_4_1`, `4_6_2`) encode an undocumented ordering and three
flex patches sit on bases `BODY_MAP` fuses into `palm`. Also confirmed the two
hands share the same 16 actuator **names**.

**Four setup bugs found, in order:**
1. No pregrip settle → the rigid hand never gripped, splat read exactly zero on
   three of seven patterns. (05 holds at pregrip for 300 steps; fixed, and now
   specified in simulated seconds.)
2. **Only 2 of 16 actuator RANGES match.** `grasp_target` returns
   `lo + fraction*(hi-lo)`, so the same fraction poses the two hands differently.
   Fixed by evaluating the pattern once on the flex model and applying the
   resulting **absolute joint angles** to both. This also removes the need for
   05's `THUMB_OPPOSED_FRACTION` workaround, which existed for this reason.
3. Flex forced to dt=0.01 → unstable (see the H4 correction above).
4. **⚠ STILL OPEN: the flex hand drops the cube.** It travels
   `[0.11, 0, 0.10]` → `[-0.17, -0.15, -0.21]`, and **39 of its 43 contacts are
   skin-on-skin, only 4 involve the cube**. So the comparison currently pits a
   hand gripping an object against a hand that has let go.

**Consequence: every splat-vs-flex agreement number so far is void.** Do not
quote the −0.066 correlation. Fixing it needs a grasp tuned to hold on the flex
hand — the same work 05 had to do for the rigid hand.

#### What DOES stand: the splat is blind to fingertip contact

Independent of flex, from the rigid side alone:

```
contacts by body:  palm 4, if_ds 2, mf_ds 2, rf_ds 1, th_mp 1
taxels fired:      1 of 368, palm only
bodies WITH contact but ZERO taxels firing: if_ds, mf_ds, rf_ds, th_mp
```

All four fingertips touch the cube; none of their 120 taxels fire. This is F3/D4
finally showing a functional consequence.

**Cutoff sweep — the problem is positional, not a threshold.** Widening
`kernel_cutoff` from the reference 10 mm:

| cutoff | fingertip patches active | notes |
|---|---|---|
| 10 mm | 0 | the reference constant |
| 13 mm | 0 | still nothing |
| 16 mm | 2 | ~0.5 of 30 taxels/frame |
| 20 mm | 2 | identical to 16 mm |

So "just raise the cutoff" does not work, and would break D5 anyway (the encoder
would no longer reproduce the reference the oracle validates against). The
override is behind `--cutoff-mm` and prints a warning naming D5.

**Reframe D4.** It was logged as "affects sim-to-real fidelity, does NOT affect
throughput, bottleneck location or kernel correctness — not a blocker." That was
right for the performance work and is now insufficient: the encoder cannot see
fingertip contact on this hand, and fingertips are where in-hand manipulation
happens. **This is now the highest-value question for Hamid.**

#### Process notes, all of which cost real time today

* **Rendering earns its keep.** The flex-drops-cube bug was invisible in the
  numbers and obvious in the picture. Ask for the image earlier.
* **`pkill -f <pattern>` kills its own shell** when the invoking command line
  contains the pattern. That was the mysterious `exit 144` on three attempts.
  Use `ps -eo pid,comm,args | awk '/pat/ && $2 ~ /python/ {print $1}' | xargs kill`.
* **Pipes defeat `python -u`.** `| head` / `| tee` buffered output and made a
  working 15-minute job look like a hang. Redirect to a file.
* **Estimate runtime before setting `timeout`.** A run was killed at 14:31 of a
  900 s limit.
* **Fixed step counts are a trap once dt varies.** Settle in simulated seconds.
* Run one GPU job at a time; a contended `--tactile none` check read 4× slow and
  looked like a regression.

### 2026-08-13 — session 2

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
