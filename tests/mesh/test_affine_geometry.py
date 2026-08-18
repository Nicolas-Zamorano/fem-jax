"""Tests for CellBasis: affine geometry mapping, Jacobians, weights, values,
gradients.

See Section 10.2 of the architecture specification; GEO-01 through GEO-04
of the FEM testing plan.

Quantities constant along the cell or quadrature axis are stored at their
natural, un-repeated shape rather than broadcast out to the full
``(K, Q, ...)`` shape (see ``CellBasis``'s own docstring): ``jacobians``,
``inverse_jacobians``, ``jacobian_determinants``, and ``gradients`` collapse
their ``Q`` axis to ``1``; ``values`` collapses its ``K`` axis to ``1``.
"""

import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.reference_cell.quadrature import QuadratureRule
from jax_fem.space import create_cell_basis, create_finite_element_space

# A single skewed (non-axis-aligned) triangle, area = 0.5 * |cross((2,0),(0.5,3))| = 3.
# Its Jacobian [[2, 0.5], [0, 3]] is non-diagonal (J != J^T), so it also
# serves GEO-04's "hide a swapped transpose" requirement.
_SKEWED_TRIANGLE_VERTICES = ((0.0, 0.0), (2.0, 0.0), (0.5, 3.0))
_SKEWED_TRIANGLE_CELLS = ((0, 1, 2),)

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _make_space(vertices, cells):
    mesh = create_triangle_mesh_from_arrays(vertices, cells)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    return create_finite_element_space(mesh, element)


@pytest.mark.unit
def test_shapes() -> None:
    space = _make_space(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    k, q = 2, quadrature.points.shape[1]
    assert basis.physical_points.shape == (k, q, 1, 2)
    assert basis.jacobians.shape == (k, 1, 2, 2)
    assert basis.inverse_jacobians.shape == (k, 1, 2, 2)
    assert basis.jacobian_determinants.shape == (k, 1, 1, 1)
    assert basis.physical_weights.shape == (k, q, 1, 1)
    assert basis.values.shape == (1, q, 3, 1)
    assert basis.gradients.shape == (k, 1, 3, 2)


@pytest.mark.unit
def test_physical_points_at_reference_vertices_recover_mesh_vertices() -> None:
    """GEO-01: each reference vertex maps to the corresponding physical vertex.

    Uses a non-axis-aligned triangle so a swapped Jacobian row/column
    cannot pass accidentally.
    """
    space = _make_space(_SKEWED_TRIANGLE_VERTICES, _SKEWED_TRIANGLE_CELLS)
    reference_cell = space.element.reference_cell
    vertex_points = reference_cell.vertex_coordinates.reshape(1, 3, 1, 2)
    vertex_weights = jnp.full((1, 3, 1, 1), 1.0 / 3.0)
    quadrature = QuadratureRule(
        reference_cell=reference_cell,
        points=vertex_points,
        weights=vertex_weights,
        exactness_degree=1,
    )

    basis = create_cell_basis(space, quadrature)
    recovered = basis.physical_points[0, :, 0, :]  # (3, 2)
    expected = jnp.asarray(_SKEWED_TRIANGLE_VERTICES)
    assert jnp.allclose(recovered, expected)


@pytest.mark.unit
@pytest.mark.property
def test_forward_inverse_map_round_trip() -> None:
    """GEO-02: F_K^{-1}(F_K(xhat)) == xhat at vertices, edge points, and an
    off-center interior point.
    """
    space = _make_space(_SKEWED_TRIANGLE_VERTICES, _SKEWED_TRIANGLE_CELLS)
    reference_cell = space.element.reference_cell

    reference_points = jnp.asarray(
        [
            [
                [[0.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.5, 0.0]],
                [[0.0, 0.5]],
                [[0.5, 0.5]],
                [[0.2, 0.3]],
            ]
        ]
    )
    number_of_points = reference_points.shape[1]
    weights = jnp.full((1, number_of_points, 1, 1), 1.0 / number_of_points)
    quadrature = QuadratureRule(
        reference_cell=reference_cell,
        points=reference_points,
        weights=weights,
        exactness_degree=1,
    )
    basis = create_cell_basis(space, quadrature)

    # mesh.vertex_coordinates[...] already carries the (1, 2) row-vector
    # axis; insert only the quadrature-broadcast axis alongside it.
    origin = space.mesh.vertex_coordinates[space.mesh.cells_to_vertices[:, 0]]
    origin = origin[:, None, :, :]  # (K, 1, 1, 2)
    displacement = basis.physical_points - origin  # (K, Q, 1, 2)
    # x_row = origin_row + xhat_row @ J^T  =>  xhat_row = (x - origin)_row @ (J^{-1})^T.
    recovered_reference_points = displacement @ basis.inverse_jacobians.mT

    assert jnp.allclose(recovered_reference_points[0], reference_points[0], atol=1e-10)


@pytest.mark.unit
def test_physical_weights_sum_to_cell_area() -> None:
    space = _make_space(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(3)
    basis = create_cell_basis(space, quadrature)
    cell_areas = jnp.sum(basis.physical_weights, axis=1)[:, 0, 0]
    assert jnp.allclose(cell_areas, jnp.array([0.5, 0.5]))


@pytest.mark.unit
def test_physical_weights_sum_to_cell_area_skewed_triangle() -> None:
    space = _make_space(_SKEWED_TRIANGLE_VERTICES, _SKEWED_TRIANGLE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(1)
    basis = create_cell_basis(space, quadrature)
    cell_area = jnp.sum(basis.physical_weights)
    assert cell_area == pytest.approx(3.0)


@pytest.mark.unit
def test_jacobian_determinant_matches_twice_area_over_reference_measure() -> None:
    space = _make_space(_SKEWED_TRIANGLE_VERTICES, _SKEWED_TRIANGLE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(1)
    basis = create_cell_basis(space, quadrature)
    # area = reference_measure * |det J| => |det J| = 3 / 0.5 = 6.
    assert jnp.allclose(basis.jacobian_determinants, 6.0)


@pytest.mark.unit
def test_clockwise_triangle_is_not_rejected_but_flips_sign() -> None:
    """GEO-03: explicit orientation policy.

    Clockwise cells are not rejected (the documented "trust the caller"
    policy, ``create_triangle_mesh_from_arrays``): construction succeeds,
    but the Jacobian determinant -- and consequently ``physical_weights``
    -- comes out negative rather than being silently made positive. This
    pins the documented consequence down as tested behavior rather than
    leaving inverted-element handling unspecified.
    """
    ccw_vertices = _SKEWED_TRIANGLE_VERTICES
    cw_vertices = (ccw_vertices[0], ccw_vertices[2], ccw_vertices[1])  # swap -> CW

    ccw_space = _make_space(ccw_vertices, _SKEWED_TRIANGLE_CELLS)
    cw_space = _make_space(cw_vertices, _SKEWED_TRIANGLE_CELLS)
    quadrature = ccw_space.element.reference_cell.create_quadrature(1)

    ccw_basis = create_cell_basis(ccw_space, quadrature)
    cw_basis = create_cell_basis(cw_space, quadrature)

    assert jnp.all(ccw_basis.jacobian_determinants > 0)
    assert jnp.all(cw_basis.jacobian_determinants < 0)
    assert jnp.allclose(
        jnp.abs(ccw_basis.jacobian_determinants),
        jnp.abs(cw_basis.jacobian_determinants),
    )
    # A real, user-visible consequence: physical_weights go negative too.
    assert jnp.all(cw_basis.physical_weights < 0)


@pytest.mark.unit
def test_inverse_jacobians_are_true_inverses() -> None:
    space = _make_space(_SKEWED_TRIANGLE_VERTICES, _SKEWED_TRIANGLE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)
    identity = jnp.broadcast_to(jnp.eye(2), basis.jacobians.shape)
    assert jnp.allclose(
        basis.jacobians @ basis.inverse_jacobians, identity, atol=1e-10
    )


@pytest.mark.unit
def test_physical_gradient_transformation_independent_solve() -> None:
    """GEO-04: grad_x(phi_i) cross-checked by an independently solved linear system.

    ``grad_phys @ J = grad_ref`` (row-vector convention), solved here via
    ``jnp.linalg.solve`` -- a different code path from ``create_cell_basis``'s
    own ``jnp.linalg.inv`` -- on the skewed, non-diagonal-Jacobian triangle
    so a swapped transpose cannot pass accidentally. Both sides are a
    single constant 2x2 system, since gradients are constant for affine P1.
    """
    space = _make_space(_SKEWED_TRIANGLE_VERTICES, _SKEWED_TRIANGLE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    reference_gradients = space.element.tabulate_basis_gradients(
        quadrature.points
    )[0, 0]  # (3, 2), constant
    jacobian = basis.jacobians[0, 0]  # (2, 2)

    # grad_phys @ J = grad_ref  <=>  J^T @ grad_phys^T = grad_ref^T.
    solved_transpose = jnp.linalg.solve(jacobian.mT, reference_gradients.mT)  # (2, 3)
    expected_gradients = solved_transpose.mT  # (3, 2)

    assert jnp.allclose(basis.gradients[0, 0], expected_gradients, atol=1e-10)


@pytest.mark.unit
@pytest.mark.integration
def test_gradient_and_value_reconstruction_of_affine_function() -> None:
    space = _make_space(_SKEWED_TRIANGLE_VERTICES, _SKEWED_TRIANGLE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    a, b, c = 2.0, -3.0, 5.0

    def u(xy):
        return a * xy[..., 0] + b * xy[..., 1] + c

    dof_values = u(space.dof_coordinates)  # (N_dofs, 1)
    local_dof_values = dof_values[space.cells_to_dofs]  # (K, N_phi, 1)
    local_dof_values_row = local_dof_values[:, None, :, :]  # (K, 1, N_phi, 1)

    # Exact formulas from Section 11.2.
    values = basis.values.mT @ local_dof_values_row  # (K, Q, 1, 1)
    gradients = local_dof_values_row.mT @ basis.gradients  # (K, 1, 1, d)

    expected_values = u(basis.physical_points)  # (K, Q, 1)
    assert jnp.allclose(values[..., 0, 0], expected_values[..., 0])
    assert jnp.allclose(gradients[..., 0, :], jnp.asarray([a, b]))


@pytest.mark.unit
@pytest.mark.property
def test_values_partition_of_unity() -> None:
    space = _make_space(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
    assert jnp.allclose(jnp.sum(basis.values, axis=2), 1.0)
