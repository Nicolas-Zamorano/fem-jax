"""
Fracture transmissivities and the two-fracture benchmark problem.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jax_fem.mesh.dfn import DfnMesh
from jax_fem.problem.boundary_conditions import create_full_boundary_dirichlet_condition
from jax_fem.problem.elliptic import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
)


def evaluate_fracture_transmissivity(
    mesh: DfnMesh,
    transmissivities: ArrayLike,
    frames: jax.Array | None = None,
) -> jax.Array:
    """
    Tangential transmissivity tensor of every cell.

    Parameters
    ----------
    mesh : DfnMesh
        The fracture network mesh.
    transmissivities : ArrayLike
        Transmissivity of each fracture, in the order of
        ``mesh.fracture_tags``: shape ``(N_fractures,)`` for isotropic
        fractures, or ``(N_fractures, 2, 2)`` for tensors in the fractures'
        tangent frames.
    frames : jax.Array | None
        Tangent frame of each fracture, shape ``(N_fractures, 3, 2)``, as
        returned by ``create_fracture_frames``; required for tensor input.

    Returns
    -------
    transmissivity : jax.Array
        Tensor ``Q_i K_i Q_i^T`` of each cell's fracture ``i``, shape
        ``(K, 1, 3, 3)``.

    Raises
    ------
    ValueError
        If ``transmissivities`` has neither shape, or ``frames`` is missing
        for tensor input.

    Notes
    -----
    An isotropic transmissivity ``k_i`` gives ``k_i (I - n_i n_i^T)``, which
    equals ``Q_i (k_i I) Q_i^T`` for every orthonormal frame, so no frame is
    needed. The tensor is constant on each cell and broadcasts over the
    quadrature axis. It replaces the diffusion of evaluated coefficients,
    e.g. ``coefficients._replace(diffusion=transmissivity)``. A point
    callable cannot express it, because a point on a trace belongs to two
    fractures.
    """
    transmissivities = jnp.asarray(transmissivities)
    number_of_fractures = mesh.fracture_tags.shape[0]

    if transmissivities.shape == (number_of_fractures,):
        normals = mesh.fracture_normals
        projectors = jnp.eye(3) - normals.mT @ normals
        fracture_tensors = jnp.expand_dims(transmissivities, axis=(-2, -1)) * projectors
    elif transmissivities.shape == (number_of_fractures, 2, 2):
        if frames is None:
            raise ValueError(
                "frames is required for transmissivities of shape "
                "(N_fractures, 2, 2); see create_fracture_frames."
            )
        fracture_tensors = frames @ transmissivities @ frames.mT
    else:
        raise ValueError(
            f"transmissivities must have shape ({number_of_fractures},) or "
            f"({number_of_fractures}, 2, 2), got {transmissivities.shape}."
        )

    return jnp.expand_dims(fracture_tensors[mesh.cells_to_fractures], axis=-3)


def _dfn_two_fractures_terms(
    points: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Coordinates and profile of the two-fracture solution.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 3)``.

    Returns
    -------
    terms : tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(x, z, profile, cross_section)`` with ``profile = y (1 - y)`` and
        ``cross_section = |z| (z^2 - 1) - |x| (x^2 - 1)``, each of shape
        ``(*batch, 1)``.
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    profile = y * (1.0 - y)
    cross_section = jnp.abs(z) * (z**2 - 1.0) - jnp.abs(x) * (x**2 - 1.0)
    return x, z, profile, cross_section


def _dfn_two_fractures_solution(points: jax.Array) -> jax.Array:
    """
    Exact solution ``y (1 - y) (|z| (z^2 - 1) - |x| (x^2 - 1))``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 3)``.

    Returns
    -------
    values : jax.Array
        Solution values, shape ``(*batch, 1, 1)``.
    """
    _, _, profile, cross_section = _dfn_two_fractures_terms(points)
    return jnp.expand_dims(profile * cross_section, axis=-1)


def _dfn_two_fractures_gradient(points: jax.Array) -> jax.Array:
    """
    Gradient in ``R^3`` of the two-fracture solution.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 3)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 3)``.

    Notes
    -----
    On each fracture the component normal to it is meaningless (the
    solution has a kink across the other fracture's plane); the error
    functionals remove it with the tangent projector.
    """
    x, z, profile, cross_section = _dfn_two_fractures_terms(points)
    y = points[..., 1]
    grad_x = -profile * jnp.sign(x) * (3.0 * x**2 - 1.0)
    grad_y = (1.0 - 2.0 * y) * cross_section
    grad_z = profile * jnp.sign(z) * (3.0 * z**2 - 1.0)
    return jnp.stack((grad_x, grad_y, grad_z), axis=-1)


def _dfn_two_fractures_source(points: jax.Array) -> jax.Array:
    """
    Source ``-Lap_Gamma(u)`` of the two-fracture solution.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 3)``.

    Returns
    -------
    values : jax.Array
        Source values, shape ``(*batch, 1, 1)``.
    """
    x, z, profile, _ = _dfn_two_fractures_terms(points)
    values = (
        6.0 * profile * (jnp.abs(x) - jnp.abs(z))
        - 2.0 * jnp.abs(x) * (x**2 - 1.0)
        + 2.0 * jnp.abs(z) * (z**2 - 1.0)
    )
    return jnp.expand_dims(values, axis=-1)


def create_dfn_two_fractures_problem(mesh: DfnMesh) -> EllipticProblem:
    """
    Benchmark problem on two orthogonal fractures with a kink at their trace.

    Parameters
    ----------
    mesh : DfnMesh
        Mesh of ``F1 = (-1, 1) x (0, 1) x {0}`` and
        ``F2 = {0} x (0, 1) x (-1, 1)`` (see
        ``create_dfn_two_fractures_mesh``); its boundary tags carry the
        Dirichlet data.

    Returns
    -------
    problem : EllipticProblem
        ``-Lap_Gamma(u) = q`` with ``K = I``, ``beta = 0``, ``c = 0`` for
        ``u = y (1 - y) (|z| (z^2 - 1) - |x| (x^2 - 1))``.

    Notes
    -----
    The solution is ``-y (1 - y) |x| (x^2 - 1)`` on ``F1`` and
    ``y (1 - y) |z| (z^2 - 1)`` on ``F2``; each term vanishes on the other
    fracture, so one formula in ``R^3`` serves both. It vanishes on the
    outer fracture edges and on the trace ``x = z = 0``. Each fracture alone
    has a kink at the trace: the outward conormal derivatives of the two
    halves of ``F1`` sum to ``-2 y (1 - y)`` and those of ``F2`` to
    ``+2 y (1 - y)``, so the flux balance at the trace holds only when the
    fractures are coupled. The Dirichlet data are the exact solution.
    """
    return EllipticProblem(
        name="dfn_two_fractures",
        diffusion=constant_tensor_coefficient(jnp.eye(3)),
        advection=constant_vector_coefficient(jnp.zeros(3)),
        reaction=constant_scalar_coefficient(0.0),
        source=_dfn_two_fractures_source,
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, _dfn_two_fractures_solution),
        ),
        exact_solution=_dfn_two_fractures_solution,
        exact_gradient=_dfn_two_fractures_gradient,
    )
