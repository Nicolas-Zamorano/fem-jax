"""Tests for the lowest-order Raviart-Thomas RT0 element.

See Section 5.2 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax
import jax.numpy as jnp
import pytest

from jax_fem.element import RaviartThomasTriangleRT0
from jax_fem.reference_cell import ReferenceTriangle


@pytest.fixture
def rt0_element() -> RaviartThomasTriangleRT0:
    return RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())


@pytest.mark.unit
def test_rt0_metadata(rt0_element) -> None:
    assert rt0_element.number_of_local_dofs == 3
    assert rt0_element.value_shape == (2,)
    assert rt0_element.polynomial_degree == 0
    assert rt0_element.is_nodal is False
    assert rt0_element.mapping == "contravariant_piola"
    assert rt0_element.is_orientation_dependent is True
    assert rt0_element.entity_dofs == {
        0: ((), (), ()),
        1: ((0,), (1,), (2,)),
        2: ((),),
    }


@pytest.mark.unit
def test_rt0_basis_values_shape(rt0_element) -> None:
    points = jnp.zeros((1, 5, 1, 2))
    values = rt0_element.tabulate_basis_values(points)
    assert values.shape == (1, 5, 3, 2)


@pytest.mark.unit
def test_rt0_exact_basis_values_at_a_point(rt0_element) -> None:
    """REF-05-style: reproduces psi_0=(x,y), psi_1=(x-1,y), psi_2=(x,y-1)
    against independently hand-computed values."""
    x, y = 0.3, 0.4
    point = jnp.asarray([[[[x, y]]]])
    expected = jnp.asarray([[x, y], [x - 1.0, y], [x, y - 1.0]]).reshape(1, 1, 3, 2)
    assert jnp.allclose(
        rt0_element.tabulate_basis_values(point), expected, atol=1e-14
    )


@pytest.mark.unit
def test_rt0_normal_flux_moment_matches_facet(rt0_element) -> None:
    """Each psi_i has unit *integrated* outward normal flux through facet i
    and zero through the other two -- the defining RT0 nodal (moment)
    functional, ``int_e psi_i . n_e ds = delta_ei``. RT0's flux is constant
    along each straight reference edge, so the integral is exactly the
    midpoint's normal component times the edge length -- not the bare
    pointwise dot product, which the reference edges' differing lengths
    (hypotenuse sqrt(2), the other two 1) would NOT satisfy the Kronecker
    property by itself.
    """
    reference_cell = rt0_element.reference_cell
    facet_midpoints = jnp.asarray(
        [[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]
    ).reshape(1, 3, 1, 2)
    values = rt0_element.tabulate_basis_values(facet_midpoints)  # (1, 3, 3, 2)
    normals = reference_cell.facet_normals  # (3, 2)
    facet_lengths = jnp.asarray([2.0**0.5, 1.0, 1.0])  # facets 0, 1, 2

    # moment[e, i] = (psi_i(midpoint_e) . normal_e) * length_e
    moment = jnp.stack(
        [
            jnp.einsum("id,d->i", values[0, e], normals[e]) * facet_lengths[e]
            for e in range(3)
        ]
    )  # (facet, basis_index)
    assert jnp.allclose(moment, jnp.eye(3), atol=1e-13)


@pytest.mark.unit
@pytest.mark.jax
def test_rt0_divergence_matches_autodiff(rt0_element) -> None:
    def psi(point: jax.Array, index: int) -> jax.Array:
        return rt0_element.tabulate_basis_values(point.reshape(1, 1, 1, 2))[0, 0, index]

    key = jax.random.PRNGKey(11)
    points = jax.random.uniform(key, (5, 2))
    tabulated = rt0_element.tabulate_basis_divergences(
        points.reshape(1, 5, 1, 2)
    )  # (1, 1, 3, 1): constant, broadcasts against all 5 points

    for index in range(3):
        for point in points:
            jacobian = jax.jacobian(lambda p: psi(p, index))(point)  # (2, 2)
            autodiff_divergence = jnp.trace(jacobian)
            assert autodiff_divergence == pytest.approx(
                float(tabulated[0, 0, index, 0]), abs=1e-12
            )


@pytest.mark.unit
def test_rt0_divergence_is_constant_two(rt0_element) -> None:
    points = jnp.zeros((1, 5, 1, 2))
    divergences = rt0_element.tabulate_basis_divergences(points)
    assert divergences.shape == (1, 1, 3, 1)
    assert jnp.all(divergences == 2.0)


@pytest.mark.unit
def test_rt0_tabulate_basis_gradients_raises(rt0_element) -> None:
    points = jnp.zeros((1, 1, 1, 2))
    with pytest.raises(NotImplementedError, match="tabulate_basis_divergences"):
        rt0_element.tabulate_basis_gradients(points)


@pytest.mark.unit
def test_rt0_accepts_arbitrary_batch_size(rt0_element) -> None:
    points = jnp.zeros((5, 3, 1, 2))
    assert rt0_element.tabulate_basis_values(points).shape == (5, 3, 3, 2)
    assert rt0_element.tabulate_basis_divergences(points).shape == (5, 1, 3, 1)


@pytest.mark.unit
def test_rt0_rejects_bad_shape(rt0_element) -> None:
    with pytest.raises(ValueError):
        rt0_element.tabulate_basis_values(jnp.zeros((5, 2)))
