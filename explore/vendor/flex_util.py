"""Flex utilities: topology, displacements, and Kelvin–Voigt force estimation."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# VENDORED, UNMODIFIED, from the supervisor's `sparsh-skin-sim`:
#     source:  https://github.com/mohammad200h/sparsh-skin-sim
#     path:    util/flex_util.py
#     fetched: 2026-08-14
#
# Copied rather than reimplemented for the same reason `explore/05_grasp_motions.py`
# copies his grasp constants verbatim: this is the reference definition of what
# the flex skin reports, and comparing the splat encoder against it is only
# meaningful if it is HIS computation and not a lookalike of ours. The
# Kelvin-Voigt estimate `f = K u + B v` from flex vertex displacement is exactly
# the quantity under comparison, so it must not drift.
#
# Do not edit. If it must change, re-fetch and record the change here.
# ---------------------------------------------------------------------------

from collections import deque

import mujoco as mj
import numpy as np

def flex_id(model: mj.MjModel, flex_name: str) -> int:
    """Resolve a flex name to its model id."""
    fid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, flex_name)
    if fid < 0:
        raise ValueError(f"Flex '{flex_name}' not found")
    return int(fid)


def list_flex_names(model: mj.MjModel) -> list[str]:
    """Return all flex names in the model."""
    names: list[str] = []
    for i in range(model.nflex):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, i)
        if name is not None:
            names.append(name)
    return names


def flex_vertex_body_ids(model: mj.MjModel, flex_name: str) -> np.ndarray:
    """Body ids for every vertex of the named flex, shape (n_vert,)."""
    fid = flex_id(model, flex_name)
    adr = int(model.flex_vertadr[fid])
    n = int(model.flex_vertnum[fid])
    return np.asarray(model.flex_vertbodyid[adr : adr + n], dtype=np.int32)


def flex_joint_ids(model: mj.MjModel, flex_name: str) -> np.ndarray:
    """Joint ids owned by the flex vertex bodies, shape (n_vert, n_jnt_per_vert).

    In this hand model every vertex body has three slide joints (x, y, z).
    """
    body_ids = flex_vertex_body_ids(model, flex_name)
    if body_ids.size == 0:
        return np.zeros((0, 0), dtype=np.int32)

    jnt_counts = np.asarray(
        [int(model.body_jntnum[int(bid)]) for bid in body_ids], dtype=np.int32
    )
    if np.any(jnt_counts != jnt_counts[0]):
        raise ValueError(
            f"Flex '{flex_name}' vertices do not all have the same joint count: "
            f"{sorted(set(jnt_counts.tolist()))}"
        )

    n_jnt = int(jnt_counts[0])
    joint_ids = np.empty((body_ids.size, n_jnt), dtype=np.int32)
    for i, bid in enumerate(body_ids):
        jnt_adr = int(model.body_jntadr[int(bid)])
        if jnt_adr < 0 or n_jnt == 0:
            raise ValueError(
                f"Flex '{flex_name}' vertex body {int(bid)} has no joints"
            )
        joint_ids[i] = np.arange(jnt_adr, jnt_adr + n_jnt, dtype=np.int32)
    return joint_ids


def _joint_qpos_width(jnt_type: int) -> int:
    if jnt_type == int(mj.mjtJoint.mjJNT_FREE):
        return 7
    if jnt_type == int(mj.mjtJoint.mjJNT_BALL):
        return 4
    return 1  # slide or hinge


def flex_joint_displacements(
    model: mj.MjModel,
    data: mj.MjData,
    flex_name: str,
) -> np.ndarray:
    """Displacement of flex joints from their natural (qpos0) state.

    Parameters
    ----------
    model, data:
        Loaded MuJoCo model and corresponding data.
    flex_name:
        Name of the flex (e.g. ``\"flex_if_tip\"``).

    Returns
    -------
    displacements : ndarray, shape (n_vert, n_jnt_per_vert)
        ``data.qpos[joint] - model.qpos0[joint]`` for each vertex joint.
        For this scene every entry is a slide-joint scalar, so the array is
        typically ``(n_vert, 3)`` — xyz offset of each vertex from rest.
    """
    joint_ids = flex_joint_ids(model, flex_name)
    flat_ids = joint_ids.reshape(-1)
    for jid in flat_ids:
        width = _joint_qpos_width(int(model.jnt_type[int(jid)]))
        if width != 1:
            raise ValueError(
                f"Flex '{flex_name}' joint {int(jid)} has qpos width {width}; "
                "expected scalar slide/hinge joints"
            )

    qadr = np.asarray(model.jnt_qposadr[flat_ids], dtype=np.int32)
    displacements = np.asarray(data.qpos[qadr] - model.qpos0[qadr], dtype=np.float64)
    return displacements.reshape(joint_ids.shape)


def _with_natural_pose(model: mj.MjModel, data: mj.MjData, fn):
    """Run ``fn`` while ``data`` is at ``qpos0``, then restore the prior state."""
    qpos_saved = np.array(data.qpos, copy=True)
    qvel_saved = np.array(data.qvel, copy=True)
    try:
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)
        return fn()
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        mj.mj_forward(model, data)


def flex_rest_vertex_positions(
    model: mj.MjModel,
    data: mj.MjData,
    flex_name: str,
) -> np.ndarray:
    """World-frame vertex positions at the natural pose, shape (n_vert, 3)."""
    fid = flex_id(model, flex_name)
    adr = int(model.flex_vertadr[fid])
    n = int(model.flex_vertnum[fid])

    def _read() -> np.ndarray:
        return np.array(data.flexvert_xpos[adr : adr + n], copy=True)

    return _with_natural_pose(model, data, _read)


def all_flex_rest_vertex_positions(
    model: mj.MjModel,
    data: mj.MjData,
    flex_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Natural-pose world positions for each flex, in one forward pass."""
    names = list(flex_names) if flex_names is not None else list_flex_names(model)

    def _read() -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name in names:
            fid = flex_id(model, name)
            adr = int(model.flex_vertadr[fid])
            n = int(model.flex_vertnum[fid])
            out[name] = np.array(data.flexvert_xpos[adr : adr + n], copy=True)
        return out

    return _with_natural_pose(model, data, _read)



# ---------------------------------------------------------------------------
# Force from flex displacements (Kelvin–Voigt: f ≈ K u + B du/dt)
# ---------------------------------------------------------------------------

# Anchor-only estimate is ~5 N/m, but least-squares fits against contact
# wrenches on this hand land near 50–100 N/m (membrane + equality stiffness).
DEFAULT_FLEX_STIFFNESS = 100.0
DEFAULT_FLEX_DAMPING = 0.05



def flex_joint_qpos_dof_addresses(
    model: mj.MjModel, flex_name: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return (qpos_adr, dof_adr, shape) for all joints of a flex."""
    joint_ids = flex_joint_ids(model, flex_name)
    flat = joint_ids.reshape(-1)
    for jid in flat:
        if _joint_qpos_width(int(model.jnt_type[int(jid)])) != 1:
            raise ValueError(
                f"Flex '{flex_name}' joint {int(jid)} is not a scalar slide/hinge"
            )
    qadr = np.asarray(model.jnt_qposadr[flat], dtype=np.int32)
    dadr = np.asarray(model.jnt_dofadr[flat], dtype=np.int32)
    return qadr, dadr, joint_ids.shape


def flex_joint_velocities(
    model: mj.MjModel,
    data: mj.MjData,
    flex_name: str,
) -> np.ndarray:
    """Slide/hinge joint velocities for a flex, shape (n_vert, n_jnt_per_vert)."""
    _, dadr, shape = flex_joint_qpos_dof_addresses(model, flex_name)
    return np.asarray(data.qvel[dadr], dtype=np.float64).reshape(shape)


def _as_coeff(coeff: float | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a scalar / per-axis / per-vertex coefficient to ``shape``."""
    arr = np.asarray(coeff, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=np.float64)
    if arr.shape == shape:
        return arr
    if arr.shape == (shape[-1],):
        return np.broadcast_to(arr, shape).copy()
    if arr.shape == (shape[0], 1) or arr.shape == (shape[0],):
        return np.broadcast_to(arr.reshape(shape[0], 1), shape).copy()
    raise ValueError(
        f"Coefficient shape {arr.shape} cannot broadcast to joint shape {shape}"
    )


def kelvin_voigt_force(
    displacement: np.ndarray,
    velocity: np.ndarray,
    stiffness: float | np.ndarray = DEFAULT_FLEX_STIFFNESS,
    damping: float | np.ndarray = DEFAULT_FLEX_DAMPING,
) -> np.ndarray:
    """Estimate force from a damped spring: ``f = K u + B v`` (Newtons)."""
    u = np.asarray(displacement, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u shape {u.shape} != v shape {v.shape}")
    k = _as_coeff(stiffness, u.shape)
    b = _as_coeff(damping, u.shape)
    return k * u + b * v


def fit_kelvin_voigt(
    displacements: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    *,
    per_axis: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of ``f = K u + B v``.

    Parameters
    ----------
    displacements, velocities, forces:
        Arrays shaped ``(T, n_vert, 3)`` or ``(n_vert, 3)``.
    per_axis:
        If True, fit independent ``(K, B)`` for x/y/z (returned as shape (3,)).
        If False, fit one shared ``(K, B)`` across all axes (returned as scalars
        wrapped in 0-d arrays).

    Returns
    -------
    stiffness, damping
    """
    u = np.asarray(displacements, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    f = np.asarray(forces, dtype=np.float64)
    if u.shape != v.shape or u.shape != f.shape:
        raise ValueError("displacements, velocities, and forces must share a shape")
    if u.ndim == 2:
        u, v, f = u[None], v[None], f[None]
    if u.ndim != 3 or u.shape[-1] != 3:
        raise ValueError(f"Expected (T, n_vert, 3), got {u.shape}")

    if per_axis:
        stiffness = np.empty(3, dtype=np.float64)
        damping = np.empty(3, dtype=np.float64)
        for axis in range(3):
            a = np.column_stack((u[..., axis].ravel(), v[..., axis].ravel()))
            kb, *_ = np.linalg.lstsq(a, f[..., axis].ravel(), rcond=None)
            stiffness[axis], damping[axis] = kb
        return stiffness, damping

    a = np.column_stack((u.ravel(), v.ravel()))
    kb, *_ = np.linalg.lstsq(a, f.ravel(), rcond=None)
    return np.asarray(kb[0]), np.asarray(kb[1])


def flex_vertex_contact_forces(
    model: mj.MjModel,
    data: mj.MjData,
    flex_name: str,
    contact_geom_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """Ground-truth contact forces on flex vertices, in joint (body) axes.

    Forces are splatted from MuJoCo contacts onto vertices (barycentric when the
    contact hits an element) and projected onto each vertex body's local axes,
    which match the three slide joints. Useful as labels for
    ``fit_kelvin_voigt``.

    Returns
    -------
    forces : ndarray, shape (n_vert, 3)
    contact_count : int
    """
    fid = flex_id(model, flex_name)
    vert_adr = int(model.flex_vertadr[fid])
    vert_num = int(model.flex_vertnum[fid])
    body_ids = flex_vertex_body_ids(model, flex_name)

    contact_geom_ids: set[int] | None = None
    if contact_geom_names is not None:
        contact_geom_ids = set()
        for name in contact_geom_names:
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"Contact geom '{name}' not found")
            contact_geom_ids.add(int(gid))

    elem_num = int(model.flex_elemnum[fid])
    elem_adr = int(model.flex_elemdataadr[fid])
    elements = model.flex_elem[elem_adr : elem_adr + 3 * elem_num].reshape(elem_num, 3)

    forces_world = np.zeros((vert_num, 3), dtype=np.float64)
    contact_count = 0
    wrench = np.zeros(6, dtype=np.float64)

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        flex_sides = (int(contact.flex[0]), int(contact.flex[1]))
        if fid not in flex_sides:
            continue
        side = flex_sides.index(fid)
        other_geom = int(contact.geom[1 - side])
        if contact_geom_ids is not None and other_geom not in contact_geom_ids:
            continue

        mj.mj_contactForce(model, data, contact_id, wrench)
        force_world = contact.frame.reshape(3, 3).T @ wrench[:3]
        if side == 0:
            force_world = -force_world

        weights = _flex_contact_vertex_weights(
            model, data, contact, side, vert_adr, vert_num, elements
        )
        for vertex_id, weight in weights:
            forces_world[vertex_id] += weight * force_world
        contact_count += 1

    forces_local = np.zeros_like(forces_world)
    for i, bid in enumerate(body_ids):
        rot = data.xmat[int(bid)].reshape(3, 3)
        forces_local[i] = rot.T @ forces_world[i]
    return forces_local, contact_count


def _flex_contact_vertex_weights(
    model: mj.MjModel,
    data: mj.MjData,
    contact: mj.MjContact,
    side: int,
    vert_adr: int,
    vert_num: int,
    elements: np.ndarray,
) -> list[tuple[int, float]]:
    vertex = int(contact.vert[side])
    if 0 <= vertex < vert_num:
        return [(vertex, 1.0)]

    element = int(contact.elem[side])
    if element < 0 or element >= len(elements):
        vertices = data.flexvert_xpos[vert_adr : vert_adr + vert_num]
        distances = np.linalg.norm(vertices - contact.pos[None, :], axis=1)
        return [(int(np.argmin(distances)), 1.0)]

    triangle = elements[element]
    points = data.flexvert_xpos[vert_adr + triangle]
    basis = np.column_stack((points[1] - points[0], points[2] - points[0]))
    offsets, *_ = np.linalg.lstsq(basis, contact.pos - points[0], rcond=None)
    weights = np.array(
        [1.0 - offsets[0] - offsets[1], offsets[0], offsets[1]], dtype=np.float64
    )
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        weights[:] = 1.0 / 3.0
    else:
        weights /= total
    return [(int(triangle[i]), float(weights[i])) for i in range(3)]


class FlexForceEstimator:
    """Online Kelvin–Voigt force estimate from a multi-frame flex history.

    Each ``update`` stores the latest displacement, estimates velocity either
    from ``data.qvel`` (preferred) or from a finite-difference window over
    recent displacements, then returns ``f = K u + B v``.
    """

    def __init__(
        self,
        model: mj.MjModel,
        flex_name: str,
        *,
        stiffness: float | np.ndarray = DEFAULT_FLEX_STIFFNESS,
        damping: float | np.ndarray = DEFAULT_FLEX_DAMPING,
        window: int = 5,
        use_qvel: bool = True,
        velocity_smoothing: float = 0.3,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.flex_name = flex_name
        self.stiffness = stiffness
        self.damping = damping
        self.window = int(window)
        self.use_qvel = bool(use_qvel)
        self.velocity_smoothing = float(np.clip(velocity_smoothing, 0.0, 1.0))

        self._qadr, self._dadr, self._shape = flex_joint_qpos_dof_addresses(
            model, flex_name
        )
        self._u_hist: deque[np.ndarray] = deque(maxlen=self.window)
        self._t_hist: deque[float] = deque(maxlen=self.window)
        self._v_smooth: np.ndarray | None = None

        # Prefer model joint damping when the caller left the default.
        if (
            isinstance(damping, (float, int))
            and float(damping) == DEFAULT_FLEX_DAMPING
        ):
            joint_ids = flex_joint_ids(model, flex_name)
            dof = model.jnt_dofadr[joint_ids.reshape(-1)]
            self.damping = float(np.mean(model.dof_damping[dof]))

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def reset(self) -> None:
        self._u_hist.clear()
        self._t_hist.clear()
        self._v_smooth = None

    def set_gains(
        self,
        stiffness: float | np.ndarray,
        damping: float | np.ndarray,
    ) -> None:
        self.stiffness = stiffness
        self.damping = damping

    def calibrate(
        self,
        displacements: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        *,
        per_axis: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit ``K, B`` from labeled batches and install them on this estimator."""
        stiffness, damping = fit_kelvin_voigt(
            displacements, velocities, forces, per_axis=per_axis
        )
        self.set_gains(stiffness, damping)
        return stiffness, damping

    def _displacement(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray:
        return np.asarray(
            data.qpos[self._qadr] - model.qpos0[self._qadr], dtype=np.float64
        ).reshape(self._shape)

    def _velocity_from_qvel(self, data: mj.MjData) -> np.ndarray:
        return np.asarray(data.qvel[self._dadr], dtype=np.float64).reshape(self._shape)

    def _velocity_from_history(self) -> np.ndarray:
        if len(self._u_hist) < 2:
            return np.zeros(self._shape, dtype=np.float64)
        u_now = self._u_hist[-1]
        u_prev = self._u_hist[-2]
        dt = self._t_hist[-1] - self._t_hist[-2]
        if dt <= 0.0:
            return np.zeros(self._shape, dtype=np.float64)
        return (u_now - u_prev) / dt

    def update(
        self, model: mj.MjModel, data: mj.MjData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ingest the current frame and return ``(force, displacement, velocity)``.

        ``force`` has shape ``(n_vert, 3)`` in Newtons, expressed in the same
        joint axes as the slide DOFs (body-local x/y/z).
        """
        u = self._displacement(model, data)
        self._u_hist.append(u.copy())
        self._t_hist.append(float(data.time))

        if self.use_qvel:
            v = self._velocity_from_qvel(data)
        else:
            v = self._velocity_from_history()

        if self._v_smooth is None or self.velocity_smoothing <= 0.0:
            self._v_smooth = v.copy()
        else:
            a = self.velocity_smoothing
            self._v_smooth = (1.0 - a) * self._v_smooth + a * v

        force = kelvin_voigt_force(
            u, self._v_smooth, self.stiffness, self.damping
        )
        return force, u, self._v_smooth.copy()

    def history_displacements(self) -> np.ndarray:
        """Stacked recent displacements, shape ``(T, n_vert, 3)``."""
        if not self._u_hist:
            return np.zeros((0, *self._shape), dtype=np.float64)
        return np.stack(list(self._u_hist), axis=0)


class AllFlexForceEstimator:
    """Kelvin–Voigt estimators for every flex in the model."""

    def __init__(
        self,
        model: mj.MjModel,
        flex_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        names = list(flex_names) if flex_names is not None else list_flex_names(model)
        self.estimators = {
            name: FlexForceEstimator(model, name, **kwargs) for name in names
        }

    def reset(self) -> None:
        for est in self.estimators.values():
            est.reset()

    def update(
        self, model: mj.MjModel, data: mj.MjData
    ) -> dict[str, np.ndarray]:
        """Return ``{flex_name: force(n_vert, 3)}`` for all tracked flexes."""
        forces: dict[str, np.ndarray] = {}
        for name, est in self.estimators.items():
            force, _, _ = est.update(model, data)
            forces[name] = force
        return forces

    def peak_force_magnitude(self, forces: dict[str, np.ndarray]) -> float:
        """Largest per-vertex |f| across all flexes."""
        peak = 0.0
        for force in forces.values():
            if force.size:
                peak = max(peak, float(np.max(np.linalg.norm(force, axis=1))))
        return peak


