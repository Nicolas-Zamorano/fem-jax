"""
Pointwise squared-error functionals for a mixed (H(div)) flux solution.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax

from jax_fem.function.evaluation import FiniteElementFunctionEvaluation
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.cell_basis import CellBasis

_SOLUTION_FIELD = "solution"


def flux_l2_error_functional(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    r"""
    Pointwise squared flux L2 error.

    Parameters
    ----------
    basis : CellBasis
        Evaluated H(div) basis of the flux (``mapping ==
        "contravariant_piola"``, e.g. RT0).
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element flux under the key ``"solution"``.
    problem : EllipticProblem
        The elliptic problem; requires ``exact_gradient``.

    Returns
    -------
    pointwise_values : jax.Array
        ``|sigma_h - sigma|^2`` at the quadrature points, shape
        ``(K, Q, 1, 1)``.

    Notes
    -----
    The exact flux is ``sigma = -A grad u``, written as the row vector
    ``-grad(u) A^T``.
    """
    if problem.exact_gradient is None:
        raise ValueError("problem.exact_gradient is required for the flux L2 error.")
    exact_gradient = problem.exact_gradient(basis.physical_points)
    diffusion = problem.diffusion(basis.physical_points)
    exact_flux = -(exact_gradient @ diffusion.mT)
    difference = fields[_SOLUTION_FIELD].values - exact_flux
    return difference @ difference.mT


def flux_divergence_l2_error_functional(
    basis: CellBasis,
    fields: Mapping[str, FiniteElementFunctionEvaluation],
    problem: EllipticProblem,
) -> jax.Array:
    r"""
    Pointwise squared flux-divergence L2 error.

    Parameters
    ----------
    basis : CellBasis
        Evaluated H(div) basis of the flux (``mapping ==
        "contravariant_piola"``, e.g. RT0).
    fields : Mapping[str, FiniteElementFunctionEvaluation]
        The finite element flux under the key ``"solution"``.
    problem : EllipticProblem
        The elliptic problem; ``source`` is used.

    Returns
    -------
    pointwise_values : jax.Array
        ``(div sigma_h - div sigma)^2`` at the quadrature points, shape
        ``(K, Q, 1, 1)``.

    Notes
    -----
    For the mixed Poisson system, ``div sigma = f`` holds exactly, so only
    ``problem.source`` is needed.
    """
    exact_divergence = problem.source(basis.physical_points)
    difference = fields[_SOLUTION_FIELD].divergence - exact_divergence
    return difference * difference
