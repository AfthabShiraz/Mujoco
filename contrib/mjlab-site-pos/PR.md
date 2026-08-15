# `site_pos_w` / `geom_pos_w` compute and discard orientations

## What

`EntityData.site_pos_w` is:

```python
@property
def site_pos_w(self):
    return self.site_pose_w[..., 0:3]
```

and `site_pose_w` is:

```python
@property
def site_pose_w(self):
    pos_w  = self.data.site_xpos[:, self.indexing.site_ids]
    mat_w  = self.data.site_xmat[:, self.indexing.site_ids]
    quat_w = quat_from_matrix(mat_w)          # <-- for every site
    return torch.cat([pos_w, quat_w], dim=-1)
```

So asking for one site's **position** gathers every site's position *and*
3x3 orientation, runs `quat_from_matrix` over all of them, allocates an
`(num_envs, num_sites, 7)` tensor, then slices three columns out and discards
the quaternions it just computed. `geom_pos_w` / `geom_quat_w` are the same
shape of thing.

## Why it matters

It scales with the robot's total site count, not with how many sites the caller
asked for. On a hand with 68 sites this is wasteful but cheap. Adding 368
tactile sensor sites takes it to 436, and it becomes the single largest cost in
the environment.

Measured on `Mjlab-LeapXELA-Cube-Reorient` (a 16-DoF hand reorienting a cube),
2048 environments, NVIDIA GB10, mjlab 1.5.3 @ `f643d24`:

| robot | as shipped | with this change |
|---|---|---|
| 436 sites | 49.92 ms/env-step | **15.07 ms** (3.31x) |
| 68 sites (stock) | 16.16 ms/env-step | **14.05 ms** (1.15x) |

The stock-robot row is the point: this is not a tactile-specific problem, it is
just most visible there. The task reads `site_pos_w` twice per step -- once for
a single site (`_palm_pos_w`) and once for four (`fingertip_positions_rel_palm`).

## Correctness

The substitution is an identity, not an approximation: slicing `[..., 0:3]` from
a concatenation whose first three columns *are* `pos_w` returns `pos_w`.

Verified rather than argued:

- `torch.equal(site_pos_w_before, site_pos_w_after)` -> **True**, max|diff| =
  `0.0`, over 64 x 435 x 3 values.
- Step-zero observations of the full environment are bitwise identical.
- Multi-step trajectories are **not** used as evidence: the environment is not
  reproducible run-to-run (domain randomisation), and unpatched-vs-unpatched
  diverges as much as unpatched-vs-patched. The identity above is the argument.

## One semantic note for reviewers

The current implementation returns a fresh tensor (from `torch.cat`); this
version returns a **view** into `data.site_xpos`. I grepped mjlab and found no
in-place writes to any of the six `*_pos_w` / `*_quat_w` properties, so this is
safe today. If you would rather guarantee it, adding `.clone()` still avoids the
`quat_from_matrix` call and the concatenation, and keeps most of the win.

`*_pose_w` is left unchanged, so callers wanting the combined 7-vector are
unaffected.

## Scope

This PR changes the four expensive cases (`site_*`, `geom_*`, which call
`quat_from_matrix`). The same `pose_w[..., 0:3]` pattern also appears in
`root_link_*`, `root_com_*`, `body_link_*` and `body_com_*`; those read
quaternions directly rather than converting, so the saving is smaller (one
gather and one concatenation). Happy to extend to all of them, or to trim this
to `site_*` only -- say which you prefer.
