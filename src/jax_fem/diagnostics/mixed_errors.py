"""
High-level error computations for the mixed (H(div)) flux solution.

See Section 5.4 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax

from jax_fem.diagnostics.errors import _integrated_norm
from jax_fem.diagnostics.mixed_norms import (
    flux_divergence_l2_error_density,
    flux_l2_error_density,
)
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.cell_basis import CellBasis


def compute_flux_l2_error(
    flux_solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    reduce: bool = True,
) -> jax.Array:
    r"""
    Absolute flux L2 error ``||sigma_h - sigma_exact||_{L2(Omega)}``.

    Parameters
    ----------
    flux_solution : FiniteElementFunction
        The mixed finite element flux solution (e.g. RT0's ``sigma``).
    basis : CellBasis
        The H(div) cell basis (``mapping == "contravariant_piola"``).
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_gradient``
        (``sigma_exact = -A grad(u_exact)``).
    reduce : bool
        Whether to sum the error over all cells (i.e., compute the global
        error) or return the cell-wise error.

    Returns
    -------
    flux_l2_error : jax.Array
        The absolute flux L2 error, shape ``()`` or ``(K,)``.
    """
    return _integrated_norm(
        flux_l2_error_density, flux_solution, basis, problem, reduce
    )


def compute_flux_divergence_l2_error(
    flux_solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    reduce: bool = True,
) -> jax.Array:
    r"""
    Absolute flux-divergence L2 error
    ``||div(sigma_h) - div(sigma_exact)||_{L2(Omega)}``.

    ``div(sigma_exact) == problem.source`` exactly for the mixed Poisson
    system (Section 5.1), so this only requires ``problem.source``.

    Parameters
    ----------
    flux_solution : FiniteElementFunction
        The mixed finite element flux solution (e.g. RT0's ``sigma``).
    basis : CellBasis
        The H(div) cell basis (``mapping == "contravariant_piola"``).
    problem : EllipticProblem
        The elliptic problem.
    reduce : bool
        Whether to sum the error over all cells (i.e., compute the global
        error) or return the cell-wise error.

    Returns
    -------
    flux_divergence_l2_error : jax.Array
        The absolute flux-divergence L2 error, shape ``()`` or ``(K,)``.
    """
    return _integrated_norm(
        flux_divergence_l2_error_density, flux_solution, basis, problem, reduce
    )
