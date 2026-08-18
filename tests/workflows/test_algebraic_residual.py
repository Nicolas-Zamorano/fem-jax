"""Condensed-system algebraic residual test.

See Section 15.2 of the architecture specification; SOL-03 of the FEM
testing plan. This is not a test of a solver supplied by the library
(Section 15.3): it only checks consistency between assembly, condensation,
a user-computed free solution, and reconstruction.
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    expand_condensed_solution,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _make_laplacian_matrix_and_vector():
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(1.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    return matrix, vector


@pytest.mark.integration
@pytest.mark.end_to_end
def test_condensed_system_residual_is_small_relative_to_problem_scale() -> None:
    """SOL-03: ||A_ff u_f - F_f|| is small relative to the matrix/solution/RHS norms.

    A relative, backward-error-style tolerance rather than an unscaled
    absolute number (per the testing plan): the residual is compared
    against ``||A_ff|| ||u_f|| + ||F_f||``, not a bare epsilon.
    """
    matrix, vector = _make_laplacian_matrix_and_vector()
    dirichlet_dofs = jnp.array([0, 1, 3])
    dirichlet_values = jnp.array([[0.0], [0.0], [0.0]])
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    residual = condensed.matrix @ free_dof_values - condensed.vector

    residual_norm = float(jnp.linalg.norm(residual))
    scale = float(
        jnp.linalg.norm(condensed.matrix.todense()) * jnp.linalg.norm(free_dof_values)
        + jnp.linalg.norm(condensed.vector)
    )
    relative_residual = residual_norm / scale

    assert relative_residual < 1e-10


@pytest.mark.integration
@pytest.mark.end_to_end
def test_original_system_residual_vanishes_only_on_free_dofs() -> None:
    """The full-system residual A u - b vanishes on free DOFs after
    reconstruction, but is not expected to vanish on constrained rows
    (those rows are exactly what condensation modifies away).
    """
    matrix, vector = _make_laplacian_matrix_and_vector()
    dirichlet_dofs = jnp.array([0, 1, 3])
    dirichlet_values = jnp.array([[0.0], [0.0], [0.0]])
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full = expand_condensed_solution(free_dof_values, condensed)

    residual = matrix @ full - vector
    assert jnp.allclose(residual[condensed.free_dofs], 0.0, atol=1e-10)
