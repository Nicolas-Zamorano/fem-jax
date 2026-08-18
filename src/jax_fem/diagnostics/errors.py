"""
High-level error and norm computations for finite element solutions.
"""

import jax
import jax.numpy as jnp

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.diagnostics.norms import (
    energy_error_density,
    h1_seminorm_error_density,
    l2_error_density,
)
from jax_fem.forms.protocols import Functional
from jax_fem.function.evaluation import evaluate_finite_element_function
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.cell_basis import CellBasis


def _integrated_norm(
    squared_density: Functional,
    solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
) -> jax.Array:
    """
    Integrate a squared-error density over the mesh and take its square root.

    Parameters
    ----------
    squared_density : Functional
        The squared-error density to integrate.
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        The cell basis.
    problem : EllipticProblem
        The elliptic problem.

    Returns
    -------
    jax.Array
        The integrated norm, shape ``()``.
    """
    fields = {"solution": evaluate_finite_element_function(solution, basis)}
    pointwise = squared_density(basis, fields, problem) 
    cellwise = integrate_cellwise(pointwise, basis.physical_weights) 
    return jnp.sqrt(jnp.sum(cellwise))


def compute_l2_error(
    solution: FiniteElementFunction, basis: CellBasis, problem: EllipticProblem
) -> jax.Array:
    """
    Absolute L2 error ``||u_h - u_exact||_{L2(Omega)}``.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        The cell basis.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_solution``.

    Returns
    -------
    l2_error : jax.Array
        The absolute L2 error, shape ``()``.
    """
    return _integrated_norm(l2_error_density, solution, basis, problem)


def compute_relative_l2_error(
    solution: FiniteElementFunction, basis: CellBasis, problem: EllipticProblem
) -> jax.Array:
    """
    Relative L2 error ``||u_h - u_exact||_{L2} / ||u_exact||_{L2}``.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        The cell basis.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_solution``.

    Returns
    -------
    relative_l2_error : jax.Array
        The relative L2 error, shape ``()``.
    """
    if problem.exact_solution is None:
        raise ValueError(
            "problem.exact_solution is required for the relative L2 error."
        )
    absolute_error = compute_l2_error(solution, basis, problem)
    exact_values = problem.exact_solution(basis.physical_points)  # (K, Q, 1, 1)
    exact_norm_squared = integrate_cellwise(
        exact_values * exact_values, basis.physical_weights
    )
    return absolute_error / jnp.sqrt(jnp.sum(exact_norm_squared))


def compute_h1_seminorm_error(
    solution: FiniteElementFunction, basis: CellBasis, problem: EllipticProblem
) -> jax.Array:
    """
    H1-seminorm error ``|u_h - u_exact|_{H1} = ||grad(u_h - u_exact)||_{L2}``.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        The cell basis.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_gradient``.

    Returns
    -------
    h1_seminorm_error : jax.Array
        The H1-seminorm error, shape ``()``.
    """
    return _integrated_norm(h1_seminorm_error_density, solution, basis, problem)


def compute_energy_error(
    solution: FiniteElementFunction, basis: CellBasis, problem: EllipticProblem
) -> jax.Array:
    """
    Energy error associated with the diffusion operator's principal part.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution.
    basis : CellBasis
        The cell basis.
    problem : EllipticProblem
        The elliptic problem, requires ``problem.exact_gradient``.

    Returns
    -------
    energy_error : jax.Array
        The energy error, shape ``()``.
    """
    return _integrated_norm(energy_error_density, solution, basis, problem)
