"""Uniform-refinement convergence study for the P1 Laplacian.

See Section 19, step 11 ("exact-error functionals and convergence
examples") of the architecture specification. Mesh refinement itself is out
of scope for the initial library (Section 2.1): each resolution below is
built directly as a fresh structured mesh, not by refining a coarser one.

Uses the ``sin(pi x) sin(pi y)`` manufactured problem from
``problem/manufactured.py`` (Section 12.1); see
``test_manufactured_problems.py`` for the continuous-level verification of
that problem's math, independent of this discrete convergence check.

SOL-04 of the FEM testing plan.
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
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_structured_unit_square_mesh
from jax_fem.problem import (
    create_poisson_sin_sin_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

pytestmark = [pytest.mark.convergence, pytest.mark.end_to_end, pytest.mark.slow]


def _solve_and_measure_errors(n: int) -> tuple[float, float]:
    mesh = create_structured_unit_square_mesh(n)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)

    problem = create_poisson_sin_sin_problem(mesh)

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


def test_uniform_refinement_convergence_rates() -> None:
    """P1 theory predicts L2 error ~ O(h^2) and H1-seminorm error ~ O(h)."""
    resolutions = (4, 8, 16)
    errors = [_solve_and_measure_errors(n) for n in resolutions]

    for (l2_coarse, h1_coarse), (l2_fine, h1_fine), n_coarse, n_fine in zip(
        errors, errors[1:], resolutions, resolutions[1:], strict=False
    ):
        h_ratio = n_fine / n_coarse  # h_coarse / h_fine
        l2_rate = math.log(l2_coarse / l2_fine) / math.log(h_ratio)
        h1_rate = math.log(h1_coarse / h1_fine) / math.log(h_ratio)
        assert l2_rate == pytest.approx(2.0, abs=0.15)
        assert h1_rate == pytest.approx(1.0, abs=0.15)


def test_errors_decrease_monotonically_with_refinement() -> None:
    resolutions = (4, 8, 16, 32)
    errors = [_solve_and_measure_errors(n) for n in resolutions]
    l2_errors = [error[0] for error in errors]
    h1_errors = [error[1] for error in errors]
    assert all(a > b for a, b in zip(l2_errors, l2_errors[1:], strict=False))
    assert all(a > b for a, b in zip(h1_errors, h1_errors[1:], strict=False))
