"""End-to-end affine patch test.

See Section 16 of the architecture specification (the typical workflow,
exercised here end to end); SOL-01 of the FEM testing plan. SOL-02
(degree-p polynomial reproduction) is a future-extension test, absent
until a higher-order element exists.
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import (
    FiniteElementFunction,
    evaluate_finite_element_function,
    interpolate_function,
)
from jax_fem.mesh import create_structured_unit_square_mesh
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space


@pytest.mark.integration
@pytest.mark.end_to_end
def test_affine_patch_test(four_triangle_center_vertex_mesh, p1_element) -> None:
    """SOL-01: geometry, P1 completeness, assembly, condensation, and
    reconstruction, together, on the four-triangle mesh with a genuine
    interior DOF. u = 1 + 2x - 3y, f = 0, A = I, g = u|_boundary; the
    discrete solution must equal u_exact at every DOF coordinate exactly
    (to solver precision), since affine functions belong to every
    supported space. A trusted user-selected solver (here,
    ``jnp.linalg.solve``) is used only to complete the workflow.
    """
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p1_element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    def u_exact(x):
        return (2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0)[..., None]

    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),  # affine u: Laplacian(u) = 0.
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, u_exact),
        ),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    assert condensed.free_dofs.shape == (1,)  # only the center vertex is free

    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full_dof_values = expand_condensed_solution(free_dof_values, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full_dof_values)

    expected = interpolate_function(u_exact, space)
    assert jnp.allclose(solution.dof_values, expected.dof_values, atol=1e-11)

    solution_evaluation = evaluate_finite_element_function(solution, basis)
    expected_values = u_exact(basis.physical_points)
    assert jnp.allclose(solution_evaluation.values, expected_values, atol=1e-11)
    expected_gradients = jnp.asarray([2.0, -3.0])
    assert jnp.allclose(solution_evaluation.gradients, expected_gradients, atol=1e-11)


@pytest.mark.integration
@pytest.mark.end_to_end
@pytest.mark.parametrize("resolution", [1, 2, 3])
def test_affine_patch_test_holds_under_refinement(resolution: int) -> None:
    """SOL-01, parameterized over refinements of the center-vertex mesh.

    Mesh refinement itself is out of scope (Section 2.1): each level is a
    fresh structured n x n mesh rather than a literal refinement of the
    four-triangle mesh, but every level still has interior free DOFs.
    """
    mesh = create_structured_unit_square_mesh(2 * resolution)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    def u_exact(x):
        return (2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0)[..., None]

    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, u_exact),
        ),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    assert condensed.free_dofs.shape[0] > 0  # every level has an interior DOF

    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full_dof_values = expand_condensed_solution(free_dof_values, condensed)

    expected = interpolate_function(u_exact, space)
    assert jnp.allclose(full_dof_values, expected.dof_values, atol=1e-10)
