"""Tests for the scalar discontinuous P0 (piecewise-constant) element.

See Section 5.2 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax.numpy as jnp
import pytest

from jax_fem.element import PiecewiseConstantTriangleP0
from jax_fem.reference_cell import ReferenceTriangle


@pytest.fixture
def p0_element() -> PiecewiseConstantTriangleP0:
    return PiecewiseConstantTriangleP0(reference_cell=ReferenceTriangle())


@pytest.mark.unit
def test_p0_basis_values_shape_and_constant(p0_element) -> None:
    points = jnp.array(
        [[[[0.1, 0.2]], [[0.5, 0.4]], [[0.9, 0.05]]]]
    )  # (1, 3, 1, 2)
    values = p0_element.tabulate_basis_values(points)
    assert values.shape == (1, 3, 1, 1)
    assert jnp.all(values == 1.0)


@pytest.mark.unit
def test_p0_basis_gradients_shape_and_zero(p0_element) -> None:
    points = jnp.zeros((1, 4, 1, 2))
    gradients = p0_element.tabulate_basis_gradients(points)
    assert gradients.shape == (1, 1, 1, 2)
    assert jnp.all(gradients == 0.0)


@pytest.mark.unit
def test_p0_metadata(p0_element) -> None:
    assert p0_element.number_of_local_dofs == 1
    assert p0_element.value_shape == (1,)
    assert p0_element.polynomial_degree == 0
    assert p0_element.is_nodal is True
    assert p0_element.mapping == "identity"
    assert p0_element.is_orientation_dependent is False
    assert p0_element.entity_dofs == {
        0: ((), (), ()),
        1: ((), (), ()),
        2: ((0,),),
    }


@pytest.mark.unit
def test_p0_accepts_arbitrary_batch_size(p0_element) -> None:
    points = jnp.zeros((5, 3, 1, 2))
    assert p0_element.tabulate_basis_values(points).shape == (5, 3, 1, 1)
    assert p0_element.tabulate_basis_gradients(points).shape == (5, 1, 1, 2)


@pytest.mark.unit
def test_p0_rejects_bad_shape(p0_element) -> None:
    with pytest.raises(ValueError):
        p0_element.tabulate_basis_values(jnp.zeros((5, 2)))
