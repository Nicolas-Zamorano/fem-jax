"""Quadrature rules on reference cells.

See Section 8.3 of the architecture specification.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax_fem.reference_cell.base import ReferenceCell


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class QuadratureRule:
    """A quadrature rule attached to one reference cell.

    Quadrature is a separate immutable data object because different
    operations (e.g. mass vs. stiffness assembly, error functionals) may
    require different exactness degrees on the same reference cell.

    Attributes
    ----------
    reference_cell:
        The reference cell this rule integrates over.
    points:
        Quadrature points in reference coordinates, shape ``(1, Q, 1, d)``.
        The leading size-1 axis mirrors the cell axis of a ``CellBasis``:
        a reference-only quantity broadcasts against it directly.
    weights:
        Quadrature weights, shape ``(1, Q, 1, 1)``. Normalized so that
        ``jnp.sum(weights) == 1``: a barycentric partition of the reference
        cell, not its geometric measure. Converting to physical integration
        weights multiplies in ``reference_cell.measure`` and the Jacobian
        determinants (Section 6.3).
    exactness_degree:
        The rule integrates every polynomial of total degree at most this
        value exactly. May exceed the degree originally requested from
        ``ReferenceCell.create_quadrature`` when no rule of that exact
        degree is tabulated.
    """

    reference_cell: ReferenceCell
    points: jax.Array
    weights: jax.Array
    exactness_degree: int

    def __post_init__(self) -> None:
        if self.points.ndim != 4 or self.points.shape[0] != 1:
            raise ValueError(
                "QuadratureRule.points must have shape (1, Q, 1, d), got "
                f"{self.points.shape}."
            )
        number_of_points = self.points.shape[1]
        if self.weights.shape != (1, number_of_points, 1, 1):
            raise ValueError(
                "QuadratureRule.weights must have shape (1, Q, 1, 1) "
                f"matching points, got {self.weights.shape} for "
                f"{number_of_points} points."
            )
        if self.exactness_degree < 0:
            raise ValueError(
                "exactness_degree must be non-negative, got "
                f"{self.exactness_degree}."
            )


# Hand-coded symmetric quadrature rules on the standard reference triangle
# conv{(0, 0), (1, 0), (0, 1)}. Points are given in Cartesian reference
# coordinates and weights are normalized to sum to 1 (Section 8.3), following
# the standard Dunavant / Hammer-Marlowe-Stroud tables (whose published
# weights already sum to 1 and are conventionally scaled by the reference
# triangle's area of 0.5 to obtain integration weights).
_TRIANGLE_RULES: dict[
    int, tuple[tuple[tuple[float, float], ...], tuple[float, ...]]
] = {
    # 1 point, exact for degree <= 1.
    1: (
        ((1.0 / 3.0, 1.0 / 3.0),),
        (1.0,),
    ),
    # 3 points, exact for degree <= 2.
    2: (
        (
            (1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0),
        ),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ),
    # 6 points, exact for degree <= 4 (also covers a degree-3 request).
    4: (
        (
            (0.091576213509771, 0.091576213509771),
            (0.816847572980459, 0.091576213509771),
            (0.091576213509771, 0.816847572980459),
            (0.445948490915965, 0.445948490915965),
            (0.108103018168070, 0.445948490915965),
            (0.445948490915965, 0.108103018168070),
        ),
        (
            0.109951743655322,
            0.109951743655322,
            0.109951743655322,
            0.223381589678011,
            0.223381589678011,
            0.223381589678011,
        ),
    ),
    # 7 points, exact for degree <= 5.
    5: (
        (
            (1.0 / 3.0, 1.0 / 3.0),
            (0.470142064105115, 0.470142064105115),
            (0.059715871789770, 0.470142064105115),
            (0.470142064105115, 0.059715871789770),
            (0.101286507323456, 0.101286507323456),
            (0.797426985353087, 0.101286507323456),
            (0.101286507323456, 0.797426985353087),
        ),
        (
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ),
    ),
}


def _select_quadrature_degree(
    available_degrees: list[int], exactness_degree: int, cell_name: str
) -> int:
    """The smallest tabulated degree at least ``exactness_degree``, or raise."""
    if exactness_degree < 1:
        raise ValueError(
            f"exactness_degree must be at least 1, got {exactness_degree}."
        )
    selected_degree = next(
        (degree for degree in available_degrees if degree >= exactness_degree),
        None,
    )
    if selected_degree is None:
        raise ValueError(
            f"No tabulated {cell_name} quadrature rule reaches exactness "
            f"degree {exactness_degree}. Available degrees: "
            f"{available_degrees}."
        )
    return selected_degree


def create_triangle_quadrature(
    reference_cell: ReferenceCell,
    exactness_degree: int,
) -> QuadratureRule:
    """Build a triangle quadrature rule exact to at least ``exactness_degree``.

    Parameters
    ----------
    reference_cell:
        The reference triangle the rule is attached to.
    exactness_degree:
        Minimum polynomial exactness degree required, at least ``1``.

    Returns
    -------
    QuadratureRule
        ``points`` has shape ``(1, Q, 1, 2)``, ``weights`` has shape
        ``(1, Q, 1, 1)`` and sums to ``1``, and ``exactness_degree`` is the
        actual exactness of the selected rule: the smallest tabulated degree
        that is at least the requested one, which may be larger than
        requested when no rule of that exact degree is tabulated.
    """
    selected_degree = _select_quadrature_degree(
        sorted(_TRIANGLE_RULES), exactness_degree, "triangle"
    )

    raw_points, raw_weights = _TRIANGLE_RULES[selected_degree]
    points = jnp.asarray(raw_points).reshape(1,-1, 1, 2)
    weights = jnp.asarray(raw_weights).reshape(1,-1, 1, 1)

    return QuadratureRule(
        reference_cell=reference_cell,
        points=points,
        weights=weights,
        exactness_degree=selected_degree,
    )


# Gauss-Legendre quadrature rules on the standard reference interval [0, 1],
# used for 1D facet quadrature (Section 4.2). An n-point Gauss-Legendre rule
# is exact to polynomial degree 2n-1; nodes/weights are the standard
# reference-interval [-1, 1] rule affinely mapped to [0, 1] (node -> (node +
# 1) / 2, weight -> weight / 2), so weights still sum to 1 (Section 8.3's
# normalization convention).
_INTERVAL_RULES: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    # 1 point, exact for degree <= 1.
    1: ((0.5,), (1.0,)),
    # 2 points, exact for degree <= 3.
    3: (
        (
            0.5 - 1.0 / (2.0 * math.sqrt(3.0)),
            0.5 + 1.0 / (2.0 * math.sqrt(3.0)),
        ),
        (0.5, 0.5),
    ),
    # 3 points, exact for degree <= 5.
    5: (
        (
            0.5 * (1.0 - math.sqrt(3.0 / 5.0)),
            0.5,
            0.5 * (1.0 + math.sqrt(3.0 / 5.0)),
        ),
        (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0),
    ),
}


def create_interval_quadrature(
    reference_cell: ReferenceCell,
    exactness_degree: int,
) -> QuadratureRule:
    """Build a reference-interval quadrature rule exact to at least
    ``exactness_degree``.

    Parameters
    ----------
    reference_cell:
        The ``ReferenceInterval`` the rule is attached to.
    exactness_degree:
        Minimum polynomial exactness degree required, at least ``1``.

    Returns
    -------
    QuadratureRule
        ``points`` has shape ``(1, Q, 1, 1)``, ``weights`` has shape
        ``(1, Q, 1, 1)`` and sums to ``1``, and ``exactness_degree`` is the
        actual exactness of the selected rule (see
        ``create_triangle_quadrature``'s docstring for the selection rule,
        identical here).
    """
    selected_degree = _select_quadrature_degree(
        sorted(_INTERVAL_RULES), exactness_degree, "interval"
    )

    raw_points, raw_weights = _INTERVAL_RULES[selected_degree]
    points = jnp.asarray(raw_points).reshape(1, -1, 1, 1)
    weights = jnp.asarray(raw_weights).reshape(1, -1, 1, 1)

    return QuadratureRule(
        reference_cell=reference_cell,
        points=points,
        weights=weights,
        exactness_degree=selected_degree,
    )
