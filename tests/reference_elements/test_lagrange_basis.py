"""Tests for the scalar P1 Lagrange element (Section 9.2 of the architecture
specification; REF-01 through REF-05 of the FEM testing plan).

Reference points have shape ``(1, Q, 1, 2)``: the leading size-1 axis
mirrors the cell axis of a ``CellBasis`` (Section 6.2). Basis gradients are
constant for this affine element, so ``tabulate_basis_gradients`` returns a
single ``(1, 1, 3, 2)`` value regardless of ``Q``, rather than one per
point; it broadcasts against ``Q``-sized quantities wherever it is used.

REF-06/REF-07 (polynomial completeness for degree > 1) are future-extension
tests, absent until a higher-order element exists.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1

_CONSTANT_GRADIENTS = jnp.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


@pytest.mark.unit
def test_basis_values_shape(p1_element: LagrangeTriangleP1) -> None:
    points = jnp.zeros((1, 5, 1, 2))
    values = p1_element.tabulate_basis_values(points)
    assert values.shape == (1, 5, 3, 1)


@pytest.mark.unit
def test_basis_gradients_shape(p1_element: LagrangeTriangleP1) -> None:
    points = jnp.zeros((1, 5, 1, 2))
    gradients = p1_element.tabulate_basis_gradients(points)
    assert gradients.shape == (1, 1, 3, 2)


@pytest.mark.unit
def test_node_count_and_placement(p1_element: LagrangeTriangleP1) -> None:
    """REF-01 (adapted): P1 has exactly 3 nodes, at the reference vertices.

    P1 has no separate "node coordinate" API distinct from the reference
    cell's vertices (Section 9.2: local DOF i is owned by reference vertex
    i), so this checks that directly rather than a dedicated nodes array.
    """
    assert p1_element.number_of_local_dofs == 3
    nodes = p1_element.reference_cell.vertex_coordinates
    assert nodes.shape == (3, 2)
    assert len({tuple(row.tolist()) for row in nodes}) == 3  # distinct
    # Every node lies in the closed reference triangle: xi>=0, eta>=0, xi+eta<=1.
    assert jnp.all(nodes >= -1e-12)
    assert jnp.all(jnp.sum(nodes, axis=-1) <= 1.0 + 1e-12)


@pytest.mark.unit
def test_nodal_property(p1_element: LagrangeTriangleP1) -> None:
    """REF-02: phi_i(reference_vertex_j) == delta_ij, basis ordering consistent."""
    reference_vertices = p1_element.reference_cell.vertex_coordinates.reshape(
        1, 3, 1, 2
    )
    values = p1_element.tabulate_basis_values(reference_vertices)  # (1, 3, 3, 1)
    assert jnp.allclose(values[0, :, :, 0], jnp.eye(3))


@pytest.mark.unit
@pytest.mark.property
def test_partition_of_unity(p1_element: LagrangeTriangleP1) -> None:
    """REF-03: sum_i phi_i(x) == 1 at deterministic and random points."""
    deterministic_points = jnp.asarray(
        [[[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[1.0 / 3, 1.0 / 3]]]]
    )
    key = jax.random.PRNGKey(0)
    random_points = jax.random.uniform(key, (1, 20, 1, 2))
    points = jnp.concatenate([deterministic_points, random_points], axis=1)

    values = p1_element.tabulate_basis_values(points)
    assert jnp.allclose(jnp.sum(values, axis=2), 1.0)


@pytest.mark.unit
@pytest.mark.property
def test_gradients_sum_to_zero(p1_element: LagrangeTriangleP1) -> None:
    """REF-04: sum_i grad(phi_i) == 0."""
    points = jnp.zeros((1, 1, 1, 2))
    gradients = p1_element.tabulate_basis_gradients(points)
    assert jnp.allclose(jnp.sum(gradients, axis=2), 0.0)


@pytest.mark.unit
def test_exact_p1_basis_values_and_gradients(p1_element: LagrangeTriangleP1) -> None:
    """REF-05: reproduces phi=(1-xi-eta, xi, eta) and grad=(-1,-1),(1,0),(0,1).

    Expected arrays are computed directly from the input coordinates here,
    independently of the element's own implementation.
    """
    points = jnp.asarray(
        [[[[0.15, 0.2]], [[0.5, 0.1]], [[0.05, 0.6]], [[1.0 / 3, 1.0 / 3]]]]
    )
    xi = points[..., 0, 0]
    eta = points[..., 0, 1]
    expected_values = jnp.stack((1.0 - xi - eta, xi, eta), axis=-1)[..., None]

    assert jnp.allclose(p1_element.tabulate_basis_values(points), expected_values)
    assert jnp.allclose(
        p1_element.tabulate_basis_gradients(points), _CONSTANT_GRADIENTS
    )


@pytest.mark.unit
@pytest.mark.jax
def test_gradients_match_autodiff_of_values(p1_element: LagrangeTriangleP1) -> None:
    def values_at_point(point: jax.Array) -> jax.Array:
        # point: (2,) -> values: (3,)
        return p1_element.tabulate_basis_values(point.reshape(1, 1, 1, 2))[0, 0, :, 0]

    key = jax.random.PRNGKey(2)
    points = jax.random.uniform(key, (11, 2))
    autodiff_gradients = jax.vmap(jax.jacobian(values_at_point))(points)  # (11, 3, 2)

    tabulated_gradients = p1_element.tabulate_basis_gradients(
        points.reshape(1, 11, 1, 2)
    )  # (1, 1, 3, 2), constant: broadcasts against every one of the 11 points.
    assert jnp.allclose(autodiff_gradients, tabulated_gradients[0, 0])


@pytest.mark.unit
def test_rejects_bad_shape(p1_element: LagrangeTriangleP1) -> None:
    with pytest.raises(ValueError):
        p1_element.tabulate_basis_values(jnp.zeros((5, 2)))
