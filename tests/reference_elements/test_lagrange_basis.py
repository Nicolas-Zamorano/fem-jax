"""Tests for the scalar P1 and P2 Lagrange elements (Section 9.2 of the
architecture specification; REF-01 through REF-07 of the FEM testing plan).

Reference points have shape ``(1, Q, 1, 2)``: the leading size-1 axis
mirrors the cell axis of a ``CellBasis`` (Section 6.2). P1's basis gradients
are constant, so its ``tabulate_basis_gradients`` returns a single
``(1, 1, 3, 2)`` value regardless of ``Q``, rather than one per point; it
broadcasts against ``Q``-sized quantities wherever it is used. P2's
gradients vary with the quadrature point, so its ``tabulate_basis_gradients``
returns the fuller ``(1, Q, 6, 2)``.

REF-06/REF-07 (polynomial completeness for degree > 1, Section 3.5 of the
implementation plan) are covered by ``test_p2_polynomial_completeness``
below.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2

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
    assert jnp.allclose(values[0, :, :, 0], jnp.eye(3), atol=1e-14, rtol=1e-12)


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
    assert jnp.allclose(jnp.sum(values, axis=2), 1.0, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.property
def test_gradients_sum_to_zero(p1_element: LagrangeTriangleP1) -> None:
    """REF-04: sum_i grad(phi_i) == 0."""
    points = jnp.zeros((1, 1, 1, 2))
    gradients = p1_element.tabulate_basis_gradients(points)
    assert jnp.allclose(jnp.sum(gradients, axis=2), 0.0, atol=1e-14)


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

    assert jnp.allclose(
        p1_element.tabulate_basis_values(points),
        expected_values,
        atol=1e-14,
        rtol=1e-12,
    )
    assert jnp.allclose(
        p1_element.tabulate_basis_gradients(points),
        _CONSTANT_GRADIENTS,
        atol=1e-14,
        rtol=1e-12,
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
    assert jnp.allclose(
        autodiff_gradients, tabulated_gradients[0, 0], atol=1e-12, rtol=1e-10
    )


@pytest.mark.unit
def test_rejects_bad_shape(p1_element: LagrangeTriangleP1) -> None:
    with pytest.raises(ValueError):
        p1_element.tabulate_basis_values(jnp.zeros((5, 2)))


# ---------------------------------------------------------------------------
# P2 (quadratic) Lagrange element
# ---------------------------------------------------------------------------

# Reference vertices (local DOFs 0, 1, 2) followed by edge midpoints (local
# DOFs 3, 4, 5): facet 0 = v1-v2 midpoint, facet 1 = v2-v0, facet 2 = v0-v1.
_P2_NODES = jnp.asarray(
    [
        [[0.0, 0.0]],
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        [[0.5, 0.5]],
        [[0.0, 0.5]],
        [[0.5, 0.0]],
    ]
).reshape(1, 6, 1, 2)


@pytest.mark.unit
def test_p2_basis_values_shape(p2_element: LagrangeTriangleP2) -> None:
    points = jnp.zeros((1, 5, 1, 2))
    values = p2_element.tabulate_basis_values(points)
    assert values.shape == (1, 5, 6, 1)


@pytest.mark.unit
def test_p2_basis_gradients_shape(p2_element: LagrangeTriangleP2) -> None:
    """Unlike P1, P2's gradients vary with the quadrature point, so this
    shape is not collapsed on the Q axis."""
    points = jnp.zeros((1, 5, 1, 2))
    gradients = p2_element.tabulate_basis_gradients(points)
    assert gradients.shape == (1, 5, 6, 2)


@pytest.mark.unit
def test_p2_entity_dofs() -> None:
    entity_dofs = LagrangeTriangleP2(reference_cell=None).entity_dofs
    assert entity_dofs == {
        0: ((0,), (1,), (2,)),
        1: ((3,), (4,), (5,)),
        2: ((),),
    }


@pytest.mark.unit
def test_p2_node_count_and_placement(p2_element: LagrangeTriangleP2) -> None:
    """REF-01 (adapted): P2 has 6 nodes -- the 3 vertices and 3 edge
    midpoints -- all within the closed reference triangle."""
    assert p2_element.number_of_local_dofs == 6
    nodes = _P2_NODES[:, :, 0, :]
    assert nodes.shape == (1, 6, 2)
    distinct = {tuple(row.tolist()) for row in nodes[0]}
    assert len(distinct) == 6
    assert jnp.all(nodes >= -1e-12)
    assert jnp.all(jnp.sum(nodes, axis=-1) <= 1.0 + 1e-12)


@pytest.mark.unit
def test_p2_nodal_property(p2_element: LagrangeTriangleP2) -> None:
    """REF-02: phi_i(node_j) == delta_ij at all 6 nodes."""
    values = p2_element.tabulate_basis_values(_P2_NODES)  # (1, 6, 6, 1)
    assert jnp.allclose(values[0, :, :, 0], jnp.eye(6), atol=1e-14, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.property
def test_p2_partition_of_unity(p2_element: LagrangeTriangleP2) -> None:
    """REF-03: sum_i phi_i(x) == 1 at deterministic and random points."""
    key = jax.random.PRNGKey(1)
    random_points = jax.random.uniform(key, (1, 20, 1, 2))
    points = jnp.concatenate([_P2_NODES, random_points], axis=1)

    values = p2_element.tabulate_basis_values(points)
    assert jnp.allclose(jnp.sum(values, axis=2), 1.0, atol=1e-13, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.property
def test_p2_gradients_sum_to_zero(p2_element: LagrangeTriangleP2) -> None:
    """REF-04: sum_i grad(phi_i) == 0, at every point (not just one, since
    P2's gradients are not constant)."""
    key = jax.random.PRNGKey(3)
    points = jax.random.uniform(key, (1, 15, 1, 2))
    gradients = p2_element.tabulate_basis_gradients(points)
    assert jnp.allclose(jnp.sum(gradients, axis=2), 0.0, atol=1e-13)


@pytest.mark.unit
def test_p2_exact_basis_values_and_gradients_at_a_point(
    p2_element: LagrangeTriangleP2,
) -> None:
    """REF-05: reproduces phi_i and grad(phi_i) against independently
    hand-computed formulas at a generic (non-nodal) point."""
    xi, eta = 0.2, 0.35
    point = jnp.asarray([[[[xi, eta]]]])

    lambda_0, lambda_1, lambda_2 = 1.0 - xi - eta, xi, eta
    expected_values = jnp.asarray(
        [
            lambda_0 * (2.0 * lambda_0 - 1.0),
            lambda_1 * (2.0 * lambda_1 - 1.0),
            lambda_2 * (2.0 * lambda_2 - 1.0),
            4.0 * lambda_1 * lambda_2,
            4.0 * lambda_2 * lambda_0,
            4.0 * lambda_0 * lambda_1,
        ]
    ).reshape(1, 1, 6, 1)
    expected_gradients = jnp.asarray(
        [
            [1.0 - 4.0 * lambda_0, 1.0 - 4.0 * lambda_0],
            [4.0 * lambda_1 - 1.0, 0.0],
            [0.0, 4.0 * lambda_2 - 1.0],
            [4.0 * lambda_2, 4.0 * lambda_1],
            [-4.0 * lambda_2, 4.0 * (lambda_0 - lambda_2)],
            [4.0 * (lambda_0 - lambda_1), -4.0 * lambda_1],
        ]
    ).reshape(1, 1, 6, 2)

    assert jnp.allclose(
        p2_element.tabulate_basis_values(point), expected_values, atol=1e-14
    )
    assert jnp.allclose(
        p2_element.tabulate_basis_gradients(point), expected_gradients, atol=1e-14
    )


@pytest.mark.unit
@pytest.mark.jax
def test_p2_gradients_match_autodiff_of_values(p2_element: LagrangeTriangleP2) -> None:
    def values_at_point(point: jax.Array) -> jax.Array:
        return p2_element.tabulate_basis_values(point.reshape(1, 1, 1, 2))[0, 0, :, 0]

    key = jax.random.PRNGKey(4)
    points = jax.random.uniform(key, (11, 2))
    autodiff_gradients = jax.vmap(jax.jacobian(values_at_point))(points)  # (11, 6, 2)

    tabulated_gradients = p2_element.tabulate_basis_gradients(
        points.reshape(1, 11, 1, 2)
    )[0]  # (11, 6, 2)
    assert jnp.allclose(
        autodiff_gradients, tabulated_gradients, atol=1e-12, rtol=1e-10
    )


@pytest.mark.unit
@pytest.mark.property
def test_p2_polynomial_completeness(p2_element: LagrangeTriangleP2) -> None:
    """REF-06/REF-07: P2 exactly reproduces every quadratic polynomial.

    Nodal-interpolates a full quadratic ``u = c0 + c1 x + c2 y + c3 x^2 +
    c4 x y + c5 y^2`` at the 6 reference nodes, then checks the resulting
    basis-function combination reproduces ``u`` exactly at points that are
    not themselves nodes.
    """
    c0, c1, c2, c3, c4, c5 = 1.3, -2.1, 0.7, 1.9, -3.4, 2.2

    def u(xy: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        return c0 + c1 * x + c2 * y + c3 * x**2 + c4 * x * y + c5 * y**2

    nodal_values = u(_P2_NODES[:, :, 0, :])  # (1, 6)

    key = jax.random.PRNGKey(5)
    evaluation_points = jax.random.uniform(key, (1, 25, 1, 2))
    basis_values = p2_element.tabulate_basis_values(evaluation_points)  # (1, 25, 6, 1)
    reconstructed = jnp.sum(basis_values[..., 0] * nodal_values[:, None, :], axis=-1)

    expected = u(evaluation_points[:, :, 0, :])
    assert jnp.allclose(reconstructed, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.unit
def test_p2_rejects_bad_shape(p2_element: LagrangeTriangleP2) -> None:
    with pytest.raises(ValueError):
        p2_element.tabulate_basis_values(jnp.zeros((5, 2)))


# ---------------------------------------------------------------------------
# Leading (batch) axis genuinely arbitrary, not just 1
# ---------------------------------------------------------------------------
#
# CellBasis always shares one reference point set across every cell (leading
# axis 1), but FacetBasis (Section 4.2) evaluates a *distinct* reference
# point per facet within that facet's own cell -- a genuine batch axis B > 1
# -- so tabulate_basis_values/tabulate_basis_gradients must work for any B,
# not just B == 1.


@pytest.mark.unit
@pytest.mark.jax
def test_p1_tabulate_accepts_arbitrary_batch_size(p1_element: LagrangeTriangleP1) -> None:
    key = jax.random.PRNGKey(6)
    points = jax.random.uniform(key, (5, 3, 1, 2))  # B=5 distinct point sets

    values = p1_element.tabulate_basis_values(points)
    gradients = p1_element.tabulate_basis_gradients(points)
    assert values.shape == (5, 3, 3, 1)
    # Gradients are constant, but must broadcast to every one of the 5
    # independent batch slots, not silently stay hardcoded to shape[0] == 1.
    assert gradients.shape == (5, 1, 3, 2)
    assert jnp.allclose(gradients, _CONSTANT_GRADIENTS, atol=1e-14)

    # Cross-check against per-slot evaluation: each batch slot is
    # independent, so evaluating one point set at a time must match.
    for b in range(5):
        one = p1_element.tabulate_basis_values(points[b : b + 1])
        assert jnp.allclose(values[b : b + 1], one, atol=1e-14)


@pytest.mark.unit
@pytest.mark.jax
def test_p2_tabulate_accepts_arbitrary_batch_size(p2_element: LagrangeTriangleP2) -> None:
    key = jax.random.PRNGKey(7)
    points = jax.random.uniform(key, (5, 3, 1, 2))

    values = p2_element.tabulate_basis_values(points)
    gradients = p2_element.tabulate_basis_gradients(points)
    assert values.shape == (5, 3, 6, 1)
    assert gradients.shape == (5, 3, 6, 2)

    for b in range(5):
        one_values = p2_element.tabulate_basis_values(points[b : b + 1])
        one_gradients = p2_element.tabulate_basis_gradients(points[b : b + 1])
        assert jnp.allclose(values[b : b + 1], one_values, atol=1e-14)
        assert jnp.allclose(gradients[b : b + 1], one_gradients, atol=1e-14)
