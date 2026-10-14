"""End-to-end patch tests and the two-fracture benchmark on fracture networks.

Within a fracture, a linear head ``u = 1 + g . x`` has zero tangential
Laplacian. At an X-crossing the conormal fluxes of the two halves of each
fracture cancel, so ``u`` solves the network problem for every ``g`` and
every pair of transmissivities. At a T-junction the terminating fracture's
flux into the trace is not balanced unless ``g`` has no component along
that fracture's in-plane direction normal to the trace.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
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
from jax_fem.function import FiniteElementFunction, interpolate_function
from jax_fem.mesh import (
    create_dfn_mesh_from_arrays,
    create_dfn_two_fractures_mesh,
    read_dfn_mesh_avs,
)
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_dfn_two_fractures_problem,
    create_full_boundary_dirichlet_condition,
    evaluate_elliptic_coefficients,
    evaluate_fracture_transmissivity,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_embedded_cell_basis, create_finite_element_space

pytestmark = [pytest.mark.integration, pytest.mark.end_to_end]

DFNGEN_MESH = Path(__file__).parents[1] / "data" / "dfngen_two_fractures_full_mesh.inp"

_ELEMENTS = {"P1": LagrangeTriangleP1, "P2": LagrangeTriangleP2}


def _linear_problem(mesh, slope: jnp.ndarray) -> EllipticProblem:
    def solution(points):
        return 1.0 + jnp.sum(points * slope, axis=-1, keepdims=True)

    def gradient(points):
        return jnp.broadcast_to(slope, points.shape)

    return EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(3)),
        advection=constant_vector_coefficient(jnp.zeros(3)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, solution),
        ),
        exact_solution=solution,
        exact_gradient=gradient,
    )


def _solve(problem, mesh, element_name: str, transmissivities=None):
    element = _ELEMENTS[element_name](reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(5)
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    if transmissivities is not None:
        coefficients = coefficients._replace(
            diffusion=evaluate_fracture_transmissivity(mesh, transmissivities)
        )
    matrix = assemble_matrix(
        integrate_cellwise(
            elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
        ),
        space,
        space,
    )
    vector = assemble_vector(
        integrate_cellwise(
            elliptic_linear_form(basis, coefficients), basis.physical_weights
        ),
        space,
    )
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    solution = FiniteElementFunction(
        space=space, dof_values=expand_condensed_solution(free_dof_values, condensed)
    )
    return solution, basis, matrix, vector, condensed


@pytest.mark.parametrize("element_name", list(_ELEMENTS))
def test_x_crossing_reproduces_any_linear_head(
    element_name: str, x_crossing_dfn_mesh
) -> None:
    mesh = x_crossing_dfn_mesh
    problem = _linear_problem(mesh, jnp.array([1.0, 2.0, 3.0]))
    solution, basis, _, _, condensed = _solve(
        problem, mesh, element_name, transmissivities=jnp.array([1.0, 10.0])
    )
    expected = interpolate_function(problem.exact_solution, solution.space)

    assert np.all(np.asarray(mesh.facet_cell_mask)[np.asarray(mesh.trace_facets)])
    assert condensed.free_dofs.shape[0] > 0
    assert np.allclose(solution.dof_values, expected.dof_values, atol=1e-12)
    assert float(compute_h1_seminorm_error(solution, basis, problem)) < 1e-12


@pytest.mark.parametrize("element_name", list(_ELEMENTS))
def test_t_junction_reproduces_balanced_linear_head(
    element_name: str, t_junction_dfn_mesh
) -> None:
    mesh = t_junction_dfn_mesh
    problem = _linear_problem(mesh, jnp.array([1.0, 2.0, 0.0]))
    solution, basis, _, _, _ = _solve(
        problem, mesh, element_name, transmissivities=jnp.array([1.0, 10.0])
    )
    expected = interpolate_function(problem.exact_solution, solution.space)
    incidences = np.asarray(mesh.facet_cell_mask).sum(axis=-1)

    assert np.all(incidences[np.asarray(mesh.trace_facets)] == 3)
    assert np.allclose(solution.dof_values, expected.dof_values, atol=1e-12)
    assert float(compute_h1_seminorm_error(solution, basis, problem)) < 1e-12


def test_t_junction_conserves_mass_for_unbalanced_linear_head(
    t_junction_dfn_mesh,
) -> None:
    mesh = t_junction_dfn_mesh
    problem = _linear_problem(mesh, jnp.array([1.0, 2.0, 3.0]))
    solution, _, matrix, vector, condensed = _solve(
        problem, mesh, "P1", transmissivities=jnp.array([1.0, 10.0])
    )
    expected = interpolate_function(problem.exact_solution, solution.space)
    reactions = np.asarray(matrix @ solution.dof_values - vector)[:, 0]

    assert np.max(np.abs(solution.dof_values - expected.dof_values)) > 1e-3
    assert np.allclose(reactions[np.asarray(condensed.free_dofs)], 0.0, atol=1e-12)
    assert np.sum(reactions[np.asarray(condensed.dirichlet_dofs)]) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize(
    ("element_name", "expected_l2", "expected_h1"),
    [("P1", 1.70e-2, 1.85e-1), ("P2", 7.94e-4, 2.74e-2)],
)
def test_two_fractures_benchmark_on_structured_mesh(
    element_name: str, expected_l2: float, expected_h1: float
) -> None:
    mesh = create_dfn_two_fractures_mesh(4)
    problem = create_dfn_two_fractures_problem(mesh)
    solution, basis, _, _, _ = _solve(problem, mesh, element_name)

    assert float(compute_l2_error(solution, basis, problem)) == pytest.approx(
        expected_l2, rel=5e-3
    )
    assert float(compute_h1_seminorm_error(solution, basis, problem)) == pytest.approx(
        expected_h1, rel=5e-3
    )


@pytest.mark.parametrize(
    ("element_name", "max_l2", "max_h1"),
    [("P1", 3.0e-2, 2.5e-1), ("P2", 2.0e-3, 4.0e-2)],
)
def test_two_fractures_benchmark_on_dfngen_mesh(
    element_name: str, max_l2: float, max_h1: float
) -> None:
    imported = read_dfn_mesh_avs(DFNGEN_MESH)
    mesh = create_dfn_mesh_from_arrays(
        np.asarray(imported.vertex_coordinates)[:, 0, :] + np.array([0.0, 0.5, 0.0]),
        imported.cells_to_vertices,
        imported.cell_tags,
    )
    problem = create_dfn_two_fractures_problem(mesh)
    solution, basis, _, _, _ = _solve(problem, mesh, element_name)

    assert float(compute_l2_error(solution, basis, problem)) < max_l2
    assert float(compute_h1_seminorm_error(solution, basis, problem)) < max_h1
