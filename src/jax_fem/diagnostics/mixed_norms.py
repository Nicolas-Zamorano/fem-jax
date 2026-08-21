"""
Pointwise error-density functionals for the mixed (H(div)) flux solution.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax

from jax_fem.function.evaluation import FiniteElementFunctionEvaluation
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.cell_basis import CellBasis

_SOLUTION_FIELD = "solution"


def flux_l2_error_density(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    r"""Pointwise squared flux L2 error density ``|sigma_h - sigma_exact|^2``.

    ``sigma_exact = -A grad(u_exact)`` (Section 5.1's first-order system),
    as a row vector: ``(A grad(u))^T = grad(u)_row @ A^T``.

    Parameters
    ----------
    basis : CellBasis
        The H(div) cell basis (``mapping == "contravariant_piola"``, e.g.
        RT0).
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element flux solution.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_gradient``.

    Returns
    -------
    flux_l2_error : jax.Array
        Pointwise squared flux L2 error density, shape ``(K, Q, 1, 1)``.
    """
    if problem.exact_gradient is None:
        raise ValueError("problem.exact_gradient is required for the flux L2 error.")
    exact_gradient = problem.exact_gradient(basis.physical_points)  # (K, Q, 1, d)
    diffusion = problem.diffusion(basis.physical_points)  # (K, Q, d, d)
    exact_flux = -(exact_gradient @ diffusion.mT)  # (K, Q, 1, d)

    difference = fields[_SOLUTION_FIELD].values - exact_flux
    return difference @ difference.mT


def flux_divergence_l2_error_density(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    r"""Pointwise squared flux-divergence L2 error density
    ``(div(sigma_h) - div(sigma_exact))^2``.

    ``div(sigma_exact) == problem.source`` exactly, for the mixed Poisson
    system with no advection/reaction (Section 5.1: the second mixed
    equation is exactly the strong-form constraint ``div(sigma) = f`` on
    the exact solution).

    Parameters
    ----------
    basis : CellBasis
        The H(div) cell basis (``mapping == "contravariant_piola"``, e.g.
        RT0).
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element flux solution.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.source``.

    Returns
    -------
    flux_divergence_l2_error : jax.Array
        Pointwise squared flux-divergence L2 error density, shape
        ``(K, Q, 1, 1)``.
    """
    exact_divergence = problem.source(basis.physical_points)  # (K, Q, 1, 1)
    difference = fields[_SOLUTION_FIELD].divergence - exact_divergence
    return difference * difference
