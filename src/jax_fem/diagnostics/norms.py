"""
Pointwise error-density functionals for L2, H1-seminorm, and energy norms.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from jax_fem.function.evaluation import FiniteElementFunctionEvaluation
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.cell_basis import CellBasis

_SOLUTION_FIELD = "solution"


def l2_error_density(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    """Pointwise squared L2 error density ``(u_h - u_exact)^2``.

    Parameters
    ----------
    basis : CellBasis
        The cell basis.
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element solution.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_solution``.

    Returns
    -------
    l2_error : jax.Array
        Pointwise squared L2 error density, shape ``(K, Q, 1, 1)``.
    """
    if problem.exact_solution is None:
        raise ValueError("problem.exact_solution is required for the L2 error.")
    exact_values = problem.exact_solution(basis.physical_points)
    difference = fields[_SOLUTION_FIELD].values - exact_values
    return difference * difference


def h1_seminorm_error_density(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    """Pointwise squared H1-seminorm error density ``|grad u_h - grad u_exact|^2``.

    Parameters
    ----------
    basis : CellBasis
        The cell basis.
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element solution.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_gradient``.

    Returns
    -------
    h1_seminorm_error : jax.Array
        Pointwise squared H1-seminorm error density, shape ``(K, Q, 1, 1)``.
    """
    if problem.exact_gradient is None:
        raise ValueError(
            "problem.exact_gradient is required for the H1 seminorm error."
        )
    exact_gradients = problem.exact_gradient(basis.physical_points)
    difference = fields[_SOLUTION_FIELD].gradients - exact_gradients
    return difference @ difference.mT


def energy_error_density(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    """
    Pointwise squared energy error density.

    ``(grad e)^T A (grad e) + (c - div(beta) / 2) e^2``, where
    ``e = u_h - u_exact``. This is the coercive energy norm associated with
    the elliptic operator: under ``A = A^T`` uniformly positive definite,
    ``c - div(beta) / 2 >= 0``, and ``e`` in ``H^1_0(Omega)`` (e.g.
    homogeneous Dirichlet data, or boundary data represented exactly by the
    discrete trace space), the boundary contribution from the advection term
    vanishes and ``a(v, v) = ||v||_E^2`` for this density.

    Parameters
    ----------
    basis : CellBasis
        The cell basis.
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element solution.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_solution``,
        ``problem.exact_gradient``, and ``problem.advection_divergence``.

    Returns
    -------
    energy_error : jax.Array
        Pointwise squared energy error density, shape ``(K, Q, 1, 1)``.
    """
    if problem.exact_solution is None:
        raise ValueError("problem.exact_solution is required for the energy error.")
    if problem.exact_gradient is None:
        raise ValueError("problem.exact_gradient is required for the energy error.")
    if problem.advection_divergence is None:
        raise ValueError(
            "problem.advection_divergence is required for the energy error."
        )

    exact_values = problem.exact_solution(basis.physical_points)
    exact_gradients = problem.exact_gradient(basis.physical_points)
    value_difference = fields[_SOLUTION_FIELD].values - exact_values
    gradient_difference = fields[_SOLUTION_FIELD].gradients - exact_gradients

    diffusion = problem.diffusion(basis.physical_points)
    diffusion_term = (
        gradient_difference
        @ diffusion
        @ jnp.swapaxes(gradient_difference, -1, -2)
    )

    reaction = problem.reaction(basis.physical_points)
    advection_divergence = problem.advection_divergence(
        basis.physical_points
    )
    symmetrized_reaction = reaction - 0.5 * advection_divergence
    reaction_term = symmetrized_reaction * (value_difference * value_difference)

    return diffusion_term + reaction_term
