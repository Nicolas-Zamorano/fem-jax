"""Tests for the cell basis of planar triangles embedded in three dimensions."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.assembly import assemble_matrix, integrate_cellwise
from jax_fem.element import (
    LagrangeTriangleP1,
    LagrangeTriangleP2,
    PiecewiseConstantTriangleP0,
    RaviartThomasTriangleRT0,
)
from jax_fem.function import evaluate_finite_element_function, interpolate_function
from jax_fem.mesh import (
    create_dfn_mesh_from_arrays,
    create_dfn_two_fractures_mesh,
    create_triangle_mesh_from_arrays,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import (
    compute_tangent_projectors,
    create_cell_basis,
    create_embedded_cell_basis,
    create_facet_basis,
    create_finite_element_space,
)

pytestmark = pytest.mark.unit

_PLANAR_VERTICES = np.array([(0.1, -0.2), (1.3, 0.1), (1.0, 1.2), (-0.3, 0.9)])
_PLANAR_CELLS = np.array([(0, 1, 2), (0, 2, 3)])

_ELEMENTS = {
    "P0": PiecewiseConstantTriangleP0,
    "P1": LagrangeTriangleP1,
    "P2": LagrangeTriangleP2,
}


def _element(name: str):
    return _ELEMENTS[name](reference_cell=ReferenceTriangle())


def _embedded_planar_mesh(rotation=None, translation=(0.0, 0.0, 0.0)):
    vertices = np.concatenate(
        (_PLANAR_VERTICES, np.zeros((_PLANAR_VERTICES.shape[0], 1))), axis=-1
    )
    if rotation is not None:
        vertices = vertices @ rotation.T
    return create_dfn_mesh_from_arrays(
        vertices + np.asarray(translation), _PLANAR_CELLS, [1, 1]
    )


def _embedded_basis(mesh, element, degree: int = 4):
    space = create_finite_element_space(mesh, element)
    return create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(degree)
    )


def _random_rotation(seed: int) -> np.ndarray:
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return rotation


@pytest.mark.parametrize("name", list(_ELEMENTS))
def test_matches_create_cell_basis_in_the_xy_plane(name: str) -> None:
    element = _element(name)
    quadrature = element.reference_cell.create_quadrature(4)
    planar = create_cell_basis(
        create_finite_element_space(
            create_triangle_mesh_from_arrays(_PLANAR_VERTICES, _PLANAR_CELLS), element
        ),
        quadrature,
    )
    embedded = _embedded_basis(_embedded_planar_mesh(), element)

    assert embedded.physical_points.shape[-1] == 3
    assert np.allclose(embedded.physical_points[..., :2], planar.physical_points)
    assert np.allclose(embedded.physical_points[..., 2], 0.0)
    assert np.allclose(embedded.physical_weights, planar.physical_weights)
    assert np.allclose(embedded.jacobian_determinants, planar.jacobian_determinants)
    assert np.allclose(embedded.values, planar.values)
    assert np.allclose(embedded.gradients[..., :2], planar.gradients)
    assert np.allclose(embedded.gradients[..., 2], 0.0)
    assert np.allclose(embedded.inverse_jacobians[..., :2], planar.inverse_jacobians)


@pytest.mark.parametrize("name", ["P1", "P2"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_matrices_are_invariant_under_rigid_motions(name: str, seed: int) -> None:
    element = _element(name)
    reference = _embedded_basis(_embedded_planar_mesh(), element)
    moved = _embedded_basis(
        _embedded_planar_mesh(_random_rotation(seed), translation=(2.0, -1.0, 0.5)),
        element,
    )

    def local_matrices(basis):
        stiffness = integrate_cellwise(
            basis.gradients @ basis.gradients.mT, basis.physical_weights
        )
        mass = integrate_cellwise(
            basis.values @ basis.values.mT, basis.physical_weights
        )
        return stiffness, mass

    for reference_matrix, moved_matrix in zip(
        local_matrices(reference), local_matrices(moved), strict=True
    ):
        assert np.allclose(moved_matrix, reference_matrix, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradients_are_tangent_and_projectors_are_orthogonal(seed: int) -> None:
    rotation = _random_rotation(seed)
    basis = _embedded_basis(_embedded_planar_mesh(rotation), _element("P2"))
    normal = np.expand_dims(rotation[:, 2], axis=0)
    projectors = compute_tangent_projectors(basis)

    assert projectors.shape == (2, 1, 3, 3)
    assert np.allclose(basis.gradients @ normal.T, 0.0)
    assert np.allclose(projectors, projectors.mT)
    assert np.allclose(projectors @ projectors, projectors)
    assert np.allclose(normal @ projectors, 0.0)
    assert np.allclose(np.trace(projectors, axis1=-2, axis2=-1), 2.0)


def test_area_scale_is_positive_for_both_windings() -> None:
    vertices = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 2.0, 0.0)])
    counter_clockwise = _embedded_basis(
        create_dfn_mesh_from_arrays(vertices, [(0, 1, 2)], [1]), _element("P1")
    )
    clockwise = _embedded_basis(
        create_dfn_mesh_from_arrays(vertices, [(0, 2, 1)], [1]), _element("P1")
    )
    expected = np.linalg.norm(
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    )

    for basis in (counter_clockwise, clockwise):
        assert float(basis.jacobian_determinants[0, 0, 0, 0]) == pytest.approx(expected)
        assert float(jnp.sum(basis.physical_weights)) == pytest.approx(0.5 * expected)


def test_total_area_of_two_fractures_network() -> None:
    basis = _embedded_basis(create_dfn_two_fractures_mesh(3), _element("P1"))
    assert float(jnp.sum(basis.physical_weights)) == pytest.approx(4.0, rel=1e-13)


@pytest.mark.parametrize("name", ["P1", "P2"])
def test_network_stiffness_is_symmetric_semidefinite_with_constant_kernel(
    name: str,
) -> None:
    basis = _embedded_basis(create_dfn_two_fractures_mesh(2), _element(name))
    local_matrices = integrate_cellwise(
        basis.gradients @ basis.gradients.mT, basis.physical_weights
    )
    matrix = np.asarray(
        assemble_matrix(local_matrices, basis.space, basis.space).todense()
    )

    assert np.allclose(matrix, matrix.T, atol=1e-13)
    assert np.allclose(matrix @ np.ones(matrix.shape[0]), 0.0, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() > -1e-12
    assert np.sum(eigenvalues < 1e-10) == 1


@pytest.mark.parametrize("name", ["P1", "P2"])
def test_tangential_gradient_of_linear_interpolant(name: str) -> None:
    slope = jnp.array([1.0, -2.0, 3.0])
    basis = _embedded_basis(create_dfn_two_fractures_mesh(2), _element(name))
    interpolant = interpolate_function(
        lambda points: 1.0 + jnp.sum(points * slope, axis=-1, keepdims=True),
        basis.space,
    )
    gradients = evaluate_finite_element_function(interpolant, basis).gradients
    expected = jnp.expand_dims(slope, axis=0) @ compute_tangent_projectors(basis)

    assert np.allclose(gradients, expected, atol=1e-12)


def test_space_on_dfn_mesh_shares_trace_dofs() -> None:
    mesh = create_dfn_two_fractures_mesh(2)
    space = create_finite_element_space(mesh, _element("P2"))
    trace_dofs = np.asarray(mesh.trace_facets) + space.entity_dof_offset[1]
    trace_cells = np.asarray(mesh.facets_to_cells)[np.asarray(mesh.trace_facets)]

    assert space.number_of_dofs == (
        mesh.vertex_coordinates.shape[0] + mesh.facets_to_vertices.shape[0]
    )
    assert space.dof_coordinates.shape == (space.number_of_dofs, 1, 3)
    for dof, cells in zip(trace_dofs, trace_cells, strict=True):
        assert np.all(np.any(np.asarray(space.cells_to_dofs)[cells] == dof, axis=-1))


def test_tangent_projectors_are_identity_for_planar_basis() -> None:
    element = _element("P1")
    basis = create_cell_basis(
        create_finite_element_space(
            create_triangle_mesh_from_arrays(_PLANAR_VERTICES, _PLANAR_CELLS), element
        ),
        element.reference_cell.create_quadrature(1),
    )
    assert np.allclose(compute_tangent_projectors(basis), np.eye(2), atol=1e-14)


def test_rejects_piola_elements() -> None:
    mesh = create_dfn_two_fractures_mesh(1)
    space = create_finite_element_space(
        mesh, RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    )
    with pytest.raises(ValueError, match="identity"):
        create_embedded_cell_basis(space, ReferenceTriangle().create_quadrature(2))


def test_facet_basis_rejects_embedded_meshes() -> None:
    basis = _embedded_basis(create_dfn_two_fractures_mesh(1), _element("P1"))
    with pytest.raises(NotImplementedError, match="planar meshes only"):
        create_facet_basis(basis, ReferenceInterval().create_quadrature(2))
