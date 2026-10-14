"""Uniform-refinement convergence on the two-fracture benchmark network.

The solution has a kink across the trace within each fracture, but it is
smooth on each half-fracture, which the structured meshes resolve, so P1
and P2 converge at the optimal rates:

    P1: ||u - u_h||_L2 = O(h^2), |u - u_h|_H1 = O(h)
    P2: ||u - u_h||_L2 = O(h^3), |u - u_h|_H1 = O(h^2)
"""

import math

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_matrix, assemble_vector, integrate_cellwise
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import compute_h1_seminorm_error, compute_l2_error
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_dfn_two_fractures_mesh
from jax_fem.problem import (
    create_dfn_two_fractures_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_embedded_cell_basis, create_finite_element_space

pytestmark = [pytest.mark.convergence, pytest.mark.end_to_end, pytest.mark.slow]

_ELEMENTS = {"P1": LagrangeTriangleP1, "P2": LagrangeTriangleP2}
_EXPECTED_RATES = {"P1": (2.0, 1.0), "P2": (3.0, 2.0)}


def _errors(element_name: str, cells_per_unit_length: int) -> tuple[float, float]:
    mesh = create_dfn_two_fractures_mesh(cells_per_unit_length)
    element = _ELEMENTS[element_name](reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(5)
    )
    problem = create_dfn_two_fractures_problem(mesh)

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    local_matrices = integrate_cellwise(
        elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
    )
    matrix = assemble_matrix(local_matrices, space, space)
    local_vectors = integrate_cellwise(
        elliptic_linear_form(basis, coefficients), basis.physical_weights
    )
    vector = assemble_vector(local_vectors, space)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    solution = FiniteElementFunction(
        space=space, dof_values=expand_condensed_solution(free_dof_values, condensed)
    )
    return (
        float(compute_l2_error(solution, basis, problem)),
        float(compute_h1_seminorm_error(solution, basis, problem)),
    )


@pytest.mark.parametrize("element_name", list(_ELEMENTS))
def test_dfn_two_fractures_convergence_rates(element_name: str) -> None:
    resolutions = (4, 8, 16)
    errors = [_errors(element_name, n) for n in resolutions]
    expected_l2_rate, expected_h1_rate = _EXPECTED_RATES[element_name]

    for (l2_coarse, h1_coarse), (l2_fine, h1_fine), n_coarse, n_fine in zip(
        errors, errors[1:], resolutions, resolutions[1:], strict=False
    ):
        refinement = math.log(n_fine / n_coarse)
        assert math.log(l2_coarse / l2_fine) / refinement == pytest.approx(
            expected_l2_rate, abs=0.15
        )
        assert math.log(h1_coarse / h1_fine) / refinement == pytest.approx(
            expected_h1_rate, abs=0.15
        )
