"""Tests for FacetBasis: interior-facet geometry and both-sided gradients.

See Section 4.2 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.function import interpolate_function
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from jax_fem.space.facet_basis import FacetBasis, create_facet_basis

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _facet_basis_for(mesh, element_factory, facet_degree: int = 4):
    element = element_factory(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
    facet_quadrature = ReferenceInterval().create_quadrature(facet_degree)
    return space, basis, create_facet_basis(basis, facet_quadrature)


@pytest.mark.unit
def test_only_interior_facets_represented(two_triangle_unit_square_mesh) -> None:
    """The 2-cell unit-square mesh has 5 facets total, 1 interior."""
    mesh = two_triangle_unit_square_mesh
    _, _, facet_basis = _facet_basis_for(mesh, LagrangeTriangleP1)
    assert facet_basis.interior_facets.shape == (1,)
    assert set(facet_basis.cells_plus.tolist()) | set(
        facet_basis.cells_minus.tolist()
    ) == {0, 1}


@pytest.mark.unit
@pytest.mark.parametrize(
    "element_factory", [LagrangeTriangleP1, LagrangeTriangleP2], ids=["p1", "p2"]
)
def test_shapes(two_triangle_unit_square_mesh, element_factory) -> None:
    mesh = two_triangle_unit_square_mesh
    space, _, facet_basis = _facet_basis_for(mesh, element_factory)
    f = facet_basis.interior_facets.shape[0]
    q = facet_basis.quadrature.points.shape[1]
    n_phi = space.element.number_of_local_dofs
    assert facet_basis.physical_points.shape == (f, q, 1, 2)
    assert facet_basis.physical_weights.shape == (f, q, 1, 1)
    assert facet_basis.facet_lengths.shape == (f, 1, 1, 1)
    assert facet_basis.normals.shape == (f, 1, 1, 2)
    assert facet_basis.gradients_plus.shape == (f, q, n_phi, 2)
    assert facet_basis.gradients_minus.shape == (f, q, n_phi, 2)


@pytest.mark.unit
def test_facet_geometry_matches_hand_computation(two_triangle_unit_square_mesh) -> None:
    """The interior facet is the (0,0)-(1,1) diagonal: length sqrt(2), every
    physical quadrature point lies exactly on it.
    """
    mesh = two_triangle_unit_square_mesh
    _, _, facet_basis = _facet_basis_for(mesh, LagrangeTriangleP1)

    assert facet_basis.facet_lengths[0, 0, 0, 0] == pytest.approx(
        2.0**0.5, abs=1e-14
    )
    points = facet_basis.physical_points[0, :, 0, :]
    assert jnp.allclose(points[:, 0], points[:, 1], atol=1e-14)  # x == y on diagonal
    assert jnp.all(points >= -1e-14)
    assert jnp.all(points <= 1.0 + 1e-14)

    # Physical weights sum to the facet's own length (arc-length measure).
    assert jnp.sum(facet_basis.physical_weights[0, :, 0, 0]) == pytest.approx(
        2.0**0.5, abs=1e-12
    )


@pytest.mark.unit
def test_normal_is_unit_perpendicular_and_points_plus_to_minus(
    four_triangle_center_vertex_mesh,
) -> None:
    mesh = four_triangle_center_vertex_mesh
    _, _, facet_basis = _facet_basis_for(mesh, LagrangeTriangleP1)
    assert facet_basis.interior_facets.shape[0] == 4  # 4 interior spoke edges

    normals = facet_basis.normals[:, 0, 0, :]  # (F, 2)
    assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0, atol=1e-14)

    facet_vertices = mesh.facets_to_vertices[facet_basis.interior_facets]
    edge_vectors = (
        mesh.vertex_coordinates[facet_vertices[:, 1], 0]
        - mesh.vertex_coordinates[facet_vertices[:, 0], 0]
    )
    tangents = edge_vectors / jnp.linalg.norm(edge_vectors, axis=-1, keepdims=True)
    assert jnp.allclose(jnp.sum(normals * tangents, axis=-1), 0.0, atol=1e-13)

    cell_centroids = jnp.mean(mesh.vertex_coordinates[mesh.cells_to_vertices], axis=1)[
        :, 0, :
    ]
    direction = (
        cell_centroids[facet_basis.cells_minus] - cell_centroids[facet_basis.cells_plus]
    )
    assert jnp.all(jnp.sum(normals * direction, axis=-1) > 0.0)


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.parametrize(
    "element_factory", [LagrangeTriangleP1, LagrangeTriangleP2], ids=["p1", "p2"]
)
def test_gradient_reconstruction_matches_affine_function(
    two_triangle_unit_square_mesh, element_factory
) -> None:
    """Both sides must reconstruct the exact same (continuous) gradient of a
    global affine function, everywhere along the shared facet.
    """
    mesh = two_triangle_unit_square_mesh
    space, _, facet_basis = _facet_basis_for(mesh, element_factory)

    a, b, c = 2.0, -3.0, 1.0

    def u(x):
        return (a * x[..., 0] + b * x[..., 1] + c)[..., None]

    function = interpolate_function(u, space)
    local_plus = function.dof_values[space.cells_to_dofs[facet_basis.cells_plus]]
    local_minus = function.dof_values[space.cells_to_dofs[facet_basis.cells_minus]]

    grad_plus = local_plus[:, None, :, :].mT @ facet_basis.gradients_plus
    grad_minus = local_minus[:, None, :, :].mT @ facet_basis.gradients_minus

    expected = jnp.asarray((a, b))
    assert jnp.allclose(grad_plus, expected, atol=1e-11, rtol=1e-10)
    assert jnp.allclose(grad_minus, expected, atol=1e-11, rtol=1e-10)
    assert jnp.allclose(grad_plus, grad_minus, atol=1e-11, rtol=1e-10)


@pytest.mark.unit
@pytest.mark.integration
def test_gradient_reconstruction_matches_quadratic_function_p2(
    two_triangle_unit_square_mesh,
) -> None:
    """A genuine (non-affine) quadratic, still exactly in P2's local space:
    the reconstructed gradient must match the analytic gradient pointwise
    (not just at one point), and agree between both sides.
    """
    mesh = two_triangle_unit_square_mesh
    space, _, facet_basis = _facet_basis_for(mesh, LagrangeTriangleP2)

    def u(x):
        return (x[..., 0] ** 2 - x[..., 1] ** 2)[..., None]

    function = interpolate_function(u, space)
    local_plus = function.dof_values[space.cells_to_dofs[facet_basis.cells_plus]]
    local_minus = function.dof_values[space.cells_to_dofs[facet_basis.cells_minus]]

    grad_plus = local_plus[:, None, :, :].mT @ facet_basis.gradients_plus
    grad_minus = local_minus[:, None, :, :].mT @ facet_basis.gradients_minus

    x = facet_basis.physical_points[..., 0, 0]
    y = facet_basis.physical_points[..., 0, 1]
    expected = jnp.stack((2.0 * x, -2.0 * y), axis=-1)[:, :, None, :]

    assert jnp.allclose(grad_plus, expected, atol=1e-10, rtol=1e-9)
    assert jnp.allclose(grad_minus, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.unit
def test_facet_basis_shape_validation() -> None:
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)
    facet_quadrature = ReferenceInterval().create_quadrature(2)
    facet_basis = create_facet_basis(basis, facet_quadrature)

    with pytest.raises(ValueError, match="normals"):
        FacetBasis(
            space=facet_basis.space,
            quadrature=facet_basis.quadrature,
            interior_facets=facet_basis.interior_facets,
            cells_plus=facet_basis.cells_plus,
            cells_minus=facet_basis.cells_minus,
            physical_points=facet_basis.physical_points,
            physical_weights=facet_basis.physical_weights,
            facet_lengths=facet_basis.facet_lengths,
            normals=jnp.zeros((1, 1, 1, 3)),  # wrong trailing dimension
            gradients_plus=facet_basis.gradients_plus,
            gradients_minus=facet_basis.gradients_minus,
        )
