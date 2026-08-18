"""Tests for reference-triangle quadrature rules.

See Section 8.3 of the architecture specification; QUA-01 of the FEM
testing plan.

QUA-01's pass condition sums weights to ``|K_hat| = 1/2``. This library
uses a different, deliberate convention (Section 6.2/8.3): weights are
normalized to sum to ``1`` (a barycentric partition, independent of the
reference cell's measure), and ``reference_cell.measure`` supplies the
``1/2`` factor separately when converting to physical integration weights.
``test_weights_sum_to_one`` and the ``reference_triangle.measure *`` factor
in ``test_exact_for_all_monomials_up_to_degree`` reflect that convention
rather than QUA-01's literal wording.
"""

import math

import jax.numpy as jnp
import pytest


def _exact_monomial_integral(i: int, j: int) -> float:
    """Exact value of int_T x^i y^j dx dy over the reference triangle."""
    return math.factorial(i) * math.factorial(j) / math.factorial(i + j + 2)


@pytest.mark.unit
@pytest.mark.parametrize("requested_degree", [1, 2, 3, 4, 5])
def test_weights_sum_to_one(requested_degree: int, reference_triangle) -> None:
    quadrature = reference_triangle.create_quadrature(requested_degree)
    assert jnp.sum(quadrature.weights) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize("requested_degree", [1, 2, 3, 4, 5])
def test_shapes(requested_degree: int, reference_triangle) -> None:
    quadrature = reference_triangle.create_quadrature(requested_degree)
    number_of_points = quadrature.points.shape[1]
    assert quadrature.points.shape == (1, number_of_points, 1, 2)
    assert quadrature.weights.shape == (1, number_of_points, 1, 1)


@pytest.mark.unit
@pytest.mark.parametrize("requested_degree", [1, 2, 3, 4, 5])
def test_selected_exactness_covers_request(
    requested_degree: int, reference_triangle
) -> None:
    quadrature = reference_triangle.create_quadrature(requested_degree)
    assert quadrature.exactness_degree >= requested_degree


@pytest.mark.unit
def test_degree_three_reuses_degree_four_rule(reference_triangle) -> None:
    quadrature = reference_triangle.create_quadrature(3)
    assert quadrature.exactness_degree == 4


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("requested_degree", [1, 2, 3, 4, 5])
def test_exact_for_all_monomials_up_to_degree(
    requested_degree: int, reference_triangle
) -> None:
    """QUA-01: every monomial of total degree <= the rule's exactness is exact."""
    quadrature = reference_triangle.create_quadrature(requested_degree)
    x = quadrature.points[0, :, 0, 0]
    y = quadrature.points[0, :, 0, 1]
    weights = quadrature.weights[0, :, 0, 0]

    for total_degree in range(requested_degree + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            numerical = reference_triangle.measure * jnp.sum(
                weights * x**i * y**j
            )
            expected = _exact_monomial_integral(i, j)
            assert numerical == pytest.approx(expected, abs=1e-12), (i, j)


@pytest.mark.unit
def test_invalid_exactness_degree_raises(reference_triangle) -> None:
    with pytest.raises(ValueError):
        reference_triangle.create_quadrature(0)


@pytest.mark.unit
def test_exactness_degree_beyond_table_raises(reference_triangle) -> None:
    with pytest.raises(ValueError):
        reference_triangle.create_quadrature(100)
