"""Tests for local integration and global matrix/vector assembly.

See Section 14 of the architecture specification; ASM-01 through ASM-03 of
the FEM testing plan.
"""

import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem.assembly import (
    assemble_bilinear_form,
    assemble_linear_form,
    integrate_cellwise,
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


def test_integrate_cellwise_manual_example() -> None:
    # K=1, Q=2, rows=2, columns=1; weights (1/4, 3/4).
    pointwise_values = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # (1,2,2,1)
    physical_weights = jnp.array([[[[0.25]], [[0.75]]]])  # (1,2,1,1)
    result = integrate_cellwise(pointwise_values, physical_weights)
    expected = jnp.array([[[0.25 * 1.0 + 0.75 * 3.0], [0.25 * 2.0 + 0.75 * 4.0]]])
    assert result.shape == (1, 2, 1)
    assert jnp.allclose(result, expected, atol=1e-14, rtol=1e-12)


def _make_laplacian_basis():
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    return create_cell_basis(space, quadrature)


def test_assemble_bilinear_form_matches_hand_derived_global_stiffness() -> None:
    """ASM-01: manual two-triangle assembly against a hand-assembled matrix."""
    basis = _make_laplacian_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)

    assert isinstance(matrix, jax_sparse.BCOO)
    assert matrix.shape == (4, 4)

    # Hand-derived by summing the two triangles' local stiffness matrices.
    expected = jnp.array(
        [
            [1.0, -0.5, 0.0, -0.5],
            [-0.5, 1.0, -0.5, 0.0],
            [0.0, -0.5, 1.0, -0.5],
            [-0.5, 0.0, -0.5, 1.0],
        ]
    )
    assert jnp.allclose(matrix.todense(), expected, atol=1e-14, rtol=1e-12)
    # Constant-function nullspace of the pure Laplacian.
    assert jnp.allclose(jnp.sum(matrix.todense(), axis=-1), 0.0, atol=1e-14)


def test_assemble_linear_form_matches_hand_derived_global_load_vector() -> None:
    """ASM-01, vector case."""
    basis = _make_laplacian_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.zeros((2, 2))),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(1.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)

    assert vector.shape == (4, 1)
    # dof 0 and dof 2 are shared by both cells, dof 1 and dof 3 by one each.
    expected = jnp.array([[1.0 / 3.0], [1.0 / 6.0], [1.0 / 3.0], [1.0 / 6.0]])
    assert jnp.allclose(vector, expected, atol=1e-14, rtol=1e-12)


def test_sparse_assembly_matches_independent_dense_scatter() -> None:
    """ASM-02 (adapted): no separate first-class dense assembly path exists
    in this library (Section 14.2), so this cross-checks
    ``assemble_bilinear_form``'s sparse construction against a plain Python
    scatter-add over the same local matrices -- written independently of
    the library's own vectorized scatter -- on a mesh where a DOF receives
    contributions from more than one cell.
    """
    basis = _make_laplacian_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)

    pointwise = elliptic_bilinear_form(basis, coefficients)
    local_matrices = integrate_cellwise(pointwise, basis.physical_weights)  # (K,3,3)
    cells_to_dofs = basis.space.cells_to_dofs
    number_of_dofs = basis.space.number_of_dofs

    dense_reference = jnp.zeros((number_of_dofs, number_of_dofs))
    for cell_index in range(cells_to_dofs.shape[0]):
        for local_i in range(3):
            for local_j in range(3):
                global_i = int(cells_to_dofs[cell_index, local_i])
                global_j = int(cells_to_dofs[cell_index, local_j])
                dense_reference = dense_reference.at[global_i, global_j].add(
                    local_matrices[cell_index, local_i, local_j]
                )

    assert jnp.allclose(matrix.todense(), dense_reference, atol=1e-14, rtol=1e-12)


def test_assembled_matrix_is_symmetric_for_symmetric_diffusion() -> None:
    """ASM-03: global diffusion symmetry and constant nullspace, before any
    Dirichlet modification.
    """
    basis = _make_laplacian_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.array([[2.0, 0.3], [0.3, 1.0]])),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    dense = matrix.todense()
    assert jnp.allclose(dense, dense.T, atol=1e-14, rtol=1e-12)
    assert jnp.allclose(jnp.sum(dense, axis=-1), 0.0, atol=1e-14)
