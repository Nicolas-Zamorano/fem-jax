"""
Error norms of finite element solutions against an exact solution.
"""

import jax
import jax.numpy as jnp

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.diagnostics.norms import (
    energy_error_functional,
    h1_seminorm_error_functional,
    l2_error_functional,
)
from jax_fem.forms.protocols import Functional
from jax_fem.function.evaluation import evaluate_finite_element_function
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.cell_basis import CellBasis


def _integrated_norm(
    functional: Functional,
    solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    reduce: bool = True,
) -> jax.Array:
    """
    Square root of an integrated squared-error functional.

    Parameters
    ----------
    functional : Functional
        Pointwise squared-error functional.
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        Evaluated basis of ``solution.space``.
    problem : EllipticProblem
        The elliptic problem.
    reduce : bool
        If ``True``, return the global norm; otherwise return the norm on
        each cell.

    Returns
    -------
    norm : jax.Array
        The norm, shape ``()`` if ``reduce`` else ``(K, 1, 1, 1)``.
    """
    fields = {"solution": evaluate_finite_element_function(solution, basis)}
    local_contributions = integrate_cellwise(
        functional(basis, fields, problem), basis.physical_weights
    )
    if reduce:
        return jnp.sqrt(jnp.sum(local_contributions))
    return jnp.sqrt(local_contributions)


def compute_l2_error(
    solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    reduce: bool = True,
) -> jax.Array:
    """
    L2 error ``||u_h - u||_{L2}``.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        Evaluated basis of ``solution.space``.
    problem : EllipticProblem
        The elliptic problem; requires ``exact_solution``.
    reduce : bool
        If ``True``, return the global error; otherwise return the error on
        each cell.

    Returns
    -------
    l2_error : jax.Array
        The L2 error, shape ``()`` if ``reduce`` else ``(K, 1, 1, 1)``.

    Notes
    -----
    A relative error is ``compute_l2_error(...)`` divided by
    ``jnp.sqrt(jnp.sum(integrate_cellwise(u**2, basis.physical_weights)))``
    with ``u = problem.exact_solution(basis.physical_points)``.
    """
    return _integrated_norm(l2_error_functional, solution, basis, problem, reduce)


def compute_h1_seminorm_error(
    solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    reduce: bool = True,
) -> jax.Array:
    """
    H1-seminorm error ``||grad(u_h - u)||_{L2}``.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        Evaluated basis of ``solution.space``.
    problem : EllipticProblem
        The elliptic problem; requires ``exact_gradient``.
    reduce : bool
        If ``True``, return the global error; otherwise return the error on
        each cell.

    Returns
    -------
    h1_seminorm_error : jax.Array
        The H1-seminorm error, shape ``()`` if ``reduce`` else
        ``(K, 1, 1, 1)``.
    """
    return _integrated_norm(
        h1_seminorm_error_functional, solution, basis, problem, reduce
    )


def compute_energy_error(
    solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    reduce: bool = True,
) -> jax.Array:
    """
    Energy-norm error of the elliptic operator.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        Evaluated basis of ``solution.space``.
    problem : EllipticProblem
        The elliptic problem; requires ``exact_solution`` and
        ``exact_gradient``.
    reduce : bool
        If ``True``, return the global error; otherwise return the error on
        each cell.

    Returns
    -------
    energy_error : jax.Array
        The energy error, shape ``()`` if ``reduce`` else ``(K, 1, 1, 1)``.

    Notes
    -----
    Integrates ``(grad e)^T A grad e + (c - div(beta) / 2) e^2`` with
    ``e = u_h - u`` (see ``energy_error_functional``).
    """
    return _integrated_norm(energy_error_functional, solution, basis, problem, reduce)
