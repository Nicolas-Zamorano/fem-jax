"""Tests for CellBasis's contravariant-Piola mapping (RT0) and its
interaction with P0.

See Section 5.2 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax.numpy as jnp
import pytest

from jax_fem.element import PiecewiseConstantTriangleP0, RaviartThomasTriangleRT0
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))

_REVERSED_ORIENTATION_VERTICES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
_REVERSED_ORIENTATION_CELLS = ((0, 1, 2), (3, 2, 1))


def _rt0_basis(vertices, cells, degree: int = 4):
    mesh = create_triangle_mesh_from_arrays(vertices, cells)
    element = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(degree)
    return mesh, space, create_cell_basis(space, quadrature)


@pytest.mark.unit
def test_rt0_cell_basis_shapes() -> None:
    mesh, space, basis = _rt0_basis(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    k = mesh.cells_to_vertices.shape[0]
    q = basis.quadrature.points.shape[1]
    assert basis.values.shape == (k, q, 3, 2)  # not K-collapsed, unlike "identity"
    assert basis.gradients is None
    assert basis.divergences.shape == (k, 1, 3, 1)  # constant reference divergence


@pytest.mark.unit
def test_p0_cell_basis_unaffected_by_mapping_generalization() -> None:
    """P0 is "identity"-mapped, so its CellBasis looks exactly like a scalar
    Lagrange element's: values collapsed on K, divergences None."""
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = PiecewiseConstantTriangleP0(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    q = basis.quadrature.points.shape[1]
    assert basis.values.shape == (1, q, 1, 1)
    assert jnp.all(basis.values == 1.0)
    assert basis.divergences is None
    assert basis.gradients.shape == (2, 1, 1, 2)
    assert jnp.all(basis.gradients == 0.0)


@pytest.mark.unit
def test_rt0_divergence_equals_reference_divergence_over_jacobian_determinant() -> (
    None
):
    """Piola divergence map: div(psi) = (1/det(J)) * ref_div, ref_div == 2."""
    mesh, space, basis = _rt0_basis(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    expected = 2.0 / basis.jacobian_determinants[:, 0, 0, 0]
    # Signs are all +1 on this mesh (checked separately in test_dof_connectivity.py);
    # divergence magnitude must match regardless.
    assert jnp.allclose(
        jnp.abs(basis.divergences[:, 0, :, 0]), expected[:, None], atol=1e-12
    )


@pytest.mark.unit
@pytest.mark.integration
def test_rt0_physical_flux_is_continuous_across_reversed_orientation_facet() -> None:
    """The key correctness property of the Piola map + sign correction
    together: a global RT0 DOF's reconstructed physical vector field is
    *exactly* continuous across a shared facet, evaluated from either
    adjacent cell, even when the two cells traverse that facet in opposite
    local directions (the reversed-orientation fixture).
    """
    mesh, space, basis = _rt0_basis(
        _REVERSED_ORIENTATION_VERTICES, _REVERSED_ORIENTATION_CELLS
    )
    element = space.element

    interior_mask = jnp.all(mesh.facets_to_cells != -1, axis=1)
    (shared_facet,) = jnp.nonzero(interior_mask)[0]
    cell_0, cell_1 = mesh.facets_to_cells[shared_facet].tolist()

    facet_vertices = mesh.facets_to_vertices[shared_facet]
    midpoint = jnp.mean(mesh.vertex_coordinates[facet_vertices, 0, :], axis=0)

    def reconstruct(cell_index: int, global_dof: int) -> jnp.ndarray:
        origin = mesh.vertex_coordinates[mesh.cells_to_vertices[cell_index, 0], 0]
        jacobian = basis.jacobians[cell_index, 0]
        inverse_jacobian = basis.inverse_jacobians[cell_index, 0]
        reference_point = ((midpoint - origin) @ inverse_jacobian.T).reshape(1, 1, 1, 2)
        reference_values = element.tabulate_basis_values(reference_point)[0, 0]  # (3,2)
        determinant = basis.jacobian_determinants[cell_index, 0, 0, 0]
        physical_values = (reference_values @ jacobian.T) / determinant
        signs = space.local_dof_signs[cell_index]
        physical_values = physical_values * signs[:, None]

        local_dofs = space.cells_to_dofs[cell_index]
        coefficients = jnp.where(local_dofs == global_dof, 1.0, 0.0)
        return coefficients @ physical_values

    local_facet_0 = int(
        jnp.nonzero(mesh.cells_to_facets[cell_0] == shared_facet)[0][0]
    )
    global_dof = int(space.cells_to_dofs[cell_0, local_facet_0])

    sigma_from_cell_0 = reconstruct(cell_0, global_dof)
    sigma_from_cell_1 = reconstruct(cell_1, global_dof)
    assert jnp.allclose(sigma_from_cell_0, sigma_from_cell_1, atol=1e-11)
    # And it's nonzero -- a degenerate all-zero "match" would be a vacuous check.
    assert jnp.linalg.norm(sigma_from_cell_0) > 1e-6
