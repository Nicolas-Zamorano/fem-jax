"""Uniform-refinement convergence study for the P2 discretization.

See Section 3.5 of the implementation plan (``.context/implementation_plan.md``).
Mirrors ``test_elliptic_convergence.py``'s P1 study; P2 theory predicts one
extra order in each norm:

    ||u - u_h||_L2 = O(h^3), |u - u_h|_H1 = O(h)^2

SOL-04 of the FEM testing plan, P2 case.
"""

import math

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import compute_h1_seminorm_error, compute_l2_error
from jax_fem.element import LagrangeTriangleP2
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import (
    EllipticProblem,
    create_poisson_cos_sin_problem,
    create_poisson_sin_sin_problem,
    create_simple_elliptic_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

pytestmark = [pytest.mark.convergence, pytest.mark.end_to_end, pytest.mark.slow]

# P2's extra approximation power reaches the asymptotic rate at coarser
# resolutions than P1 needed (test_elliptic_convergence.py starts at n=16
# for "exponential"/"simple"): all three problems below are already within
# 0.15 of their theoretical rate by n=4 -> n=8 with P2. "exponential" (a
# sharp boundary layer) and "singular" are excluded for the same reasons as
# the P1 study (pre-asymptotic / sub-optimal-by-construction, respectively).
_PROBLEM_FACTORIES = {
    "sin_sin": create_poisson_sin_sin_problem,
    "cos_sin": create_poisson_cos_sin_problem,
    "simple": create_simple_elliptic_problem,
}


def _solve_and_measure_errors(
    problem: EllipticProblem, space, quadrature
) -> tuple[float, float]:
    basis = create_cell_basis(space, quadrature)

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full_dof_values = expand_condensed_solution(free_dof_values, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full_dof_values)

    l2_error = float(compute_l2_error(solution, basis, problem))
    h1_error = float(compute_h1_seminorm_error(solution, basis, problem))
    return l2_error, h1_error


def _errors_at_resolutions(
    name: str, resolutions: tuple[int, ...]
) -> list[tuple[float, float]]:
    element = LagrangeTriangleP2(reference_cell=ReferenceTriangle())
    errors = []
    for n in resolutions:
        mesh = create_gmsh_unit_square_mesh(n)
        space = create_finite_element_space(mesh, element)
        quadrature = space.element.reference_cell.create_quadrature(5)
        problem = _PROBLEM_FACTORIES[name](mesh)
        errors.append(_solve_and_measure_errors(problem, space, quadrature))
    return errors


@pytest.mark.parametrize("name", _PROBLEM_FACTORIES)
def test_p2_uniform_refinement_convergence_rates_and_monotonicity(name: str) -> None:
    """P2 theory predicts L2 error ~ O(h^3) and H1-seminorm error ~ O(h^2).

    Checks monotonic decrease and the asymptotic rate together from the same
    three-resolution solve, rather than as two separate studies, since P2's
    faster convergence makes both properties already visible at coarse
    resolutions.
    """
    resolutions = (4, 8, 16)
    errors = _errors_at_resolutions(name, resolutions)

    l2_errors = [error[0] for error in errors]
    h1_errors = [error[1] for error in errors]
    assert all(a > b for a, b in zip(l2_errors, l2_errors[1:], strict=False))
    assert all(a > b for a, b in zip(h1_errors, h1_errors[1:], strict=False))

    for (l2_coarse, h1_coarse), (l2_fine, h1_fine), n_coarse, n_fine in zip(
        errors, errors[1:], resolutions, resolutions[1:], strict=False
    ):
        h_ratio = n_fine / n_coarse  # h_coarse / h_fine
        l2_rate = math.log(l2_coarse / l2_fine) / math.log(h_ratio)
        h1_rate = math.log(h1_coarse / h1_fine) / math.log(h_ratio)
        assert l2_rate == pytest.approx(3.0, abs=0.3)
        assert h1_rate == pytest.approx(2.0, abs=0.3)
