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
| D6 | **Task, reward, RL algorithm inherited unmodified.** The observation space is the only thing that changes, plus whatever the profiler forces. | Preserves comparability with the supervisor's baseline. |
| D7 | **Ceiling on env count is the DGX Spark's unified memory**, not a chosen number. The sweep runs to whatever N fits under a memory cap; 4096 is not assumed reachable. | See `PROJECT_LOG.md` §1.1. |

> **D7 REVISED, 2026-08-13, same day.** Measured: **4096 envs of bare physics use 0.92 GB and run at 129,682 env-steps/s.** Memory is nowhere near binding — the earlier assumption that 4096 was out of reach on the Spark was wrong *for physics*. The sweep completed all 13 rungs with zero failures.
>
> This does **not** retract §1.1: the host crash was real. But it was not caused by physics memory at 4096. Plausible causes, none yet confirmed: the full mjlab/RSL-RL training loop (policy, optimiser, rollout buffers, WandB) rather than the sim; or the same `njmax` scaling mistake made in the first harness (below), which inflates per-world constraint memory and compute by a factor of N.
>
> **Revised stance:** keep the memory cap and subprocess isolation — they cost nothing and they caught this safely — but do not scope the project around an assumed low ceiling. Re-measure the ceiling with the *training loop* attached before concluding anything about 4096-env training.

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

**Resolution:** _unresolved_

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

**Resolution:** _unresolved_
