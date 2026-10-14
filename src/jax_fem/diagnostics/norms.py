"""
Pointwise squared-error functionals for the L2, H1-seminorm, and energy errors.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax

from jax_fem.function.evaluation import FiniteElementFunctionEvaluation
from jax_fem.problem.elliptic import EllipticProblem, evaluate_elliptic_coefficients
from jax_fem.space.cell_basis import CellBasis, compute_tangent_projectors

_SOLUTION_FIELD = "solution"


def l2_error_functional(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    """
    Pointwise squared L2 error.

    Parameters
    ----------
    basis : CellBasis
        Evaluated basis of the solution.
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element solution under the key ``"solution"``.
    problem : EllipticProblem
        The elliptic problem; requires ``exact_solution``.

    Returns
    -------
    pointwise_values : jax.Array
        ``(u_h - u)^2`` at the quadrature points, shape ``(K, Q, 1, 1)``.
    """
    if problem.exact_solution is None:
        raise ValueError("problem.exact_solution is required for the L2 error.")
    exact_values = problem.exact_solution(basis.physical_points)
    difference = fields[_SOLUTION_FIELD].values - exact_values
    return difference * difference


def h1_seminorm_error_functional(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    """
    Pointwise squared H1-seminorm error.

    Parameters
    ----------
    basis : CellBasis
        Evaluated basis of the solution.
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element solution under the key ``"solution"``.
    problem : EllipticProblem
        The elliptic problem; requires ``exact_gradient``.

    Returns
    -------
    pointwise_values : jax.Array
        ``|grad u_h - grad u P|^2`` at the quadrature points, shape
        ``(K, Q, 1, 1)``.

    Notes
    -----
    ``P`` is the projector onto each cell's tangent space (see
    ``compute_tangent_projectors``): the identity on a planar mesh, and the
    removal of the normal component on cells embedded in ``R^3``, where
    ``grad u`` may be the gradient of a function on the ambient space.
    """
    if problem.exact_gradient is None:
        raise ValueError(
            "problem.exact_gradient is required for the H1 seminorm error."
        )
    exact_gradients = problem.exact_gradient(
        basis.physical_points
    ) @ compute_tangent_projectors(basis)
    difference = fields[_SOLUTION_FIELD].gradients - exact_gradients
    return difference @ difference.mT


def energy_error_functional(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    """
    Pointwise squared energy error.

    Parameters
    ----------
    basis : CellBasis
        Evaluated basis of the solution.
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element solution under the key ``"solution"``.
    problem : EllipticProblem
        The elliptic problem; requires ``exact_solution`` and
        ``exact_gradient``.

    Returns
    -------
    pointwise_values : jax.Array
        ``(grad e)^T A grad e + (c - div(beta) / 2) e^2`` with
        ``e = u_h - u`` at the quadrature points, shape ``(K, Q, 1, 1)``.

    Notes
    -----
    This is the energy norm of the elliptic operator. When ``A`` is symmetric
    uniformly positive definite, ``c - div(beta) / 2 >= 0``, and ``e`` lies in
    ``H^1_0`` (e.g. homogeneous Dirichlet data, or boundary data represented
    exactly by the discrete space), the advection boundary term vanishes and
    ``a(e, e)`` equals the integrated functional.

    The exact gradient is projected onto each cell's tangent space, as in
    ``h1_seminorm_error_functional``.
    """
    if problem.exact_solution is None:
        raise ValueError("problem.exact_solution is required for the energy error.")
    if problem.exact_gradient is None:
        raise ValueError("problem.exact_gradient is required for the energy error.")

    exact_values = problem.exact_solution(basis.physical_points)
    exact_gradients = problem.exact_gradient(
        basis.physical_points
    ) @ compute_tangent_projectors(basis)
    value_difference = fields[_SOLUTION_FIELD].values - exact_values
    gradient_difference = fields[_SOLUTION_FIELD].gradients - exact_gradients

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    diffusion_term = (
        gradient_difference @ coefficients.diffusion @ gradient_difference.mT
    )
    symmetrized_reaction = (
        coefficients.reaction - 0.5 * coefficients.advection_divergence
    )
    reaction_term = symmetrized_reaction * (value_difference * value_difference)

    return diffusion_term + reaction_term
