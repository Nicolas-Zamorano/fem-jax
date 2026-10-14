"""
Quadrature rules on reference cells.
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
    """
    A quadrature rule on a reference cell.

    Attributes
    ----------
    reference_cell : ReferenceCell
        The reference cell the rule integrates over.
    points : jax.Array
        Reference quadrature points, shape ``(1, Q, 1, d)``.
    weights : jax.Array
        Quadrature weights normalized to sum to one, shape ``(1, Q, 1, 1)``.
    exactness_degree : int
        Highest total polynomial degree integrated exactly.

    Notes
    -----
    The leading size-one axis mirrors the cell axis of a ``CellBasis``, so
    reference quantities broadcast against cellwise arrays. Weights form a
    partition of the reference cell rather than its measure; physical weights
    multiply in ``reference_cell.measure`` and the Jacobian determinants.
    ``exactness_degree`` may exceed the requested degree when no rule of
    exactly that degree is tabulated.
    """

    reference_cell: ReferenceCell
    points: jax.Array
    weights: jax.Array
    exactness_degree: int

    def __post_init__(self) -> None:
        """
        Validate the shapes of the points and weights.

        Raises
        ------
        ValueError
            If ``points`` is not ``(1, Q, 1, d)``, ``weights`` is not
            ``(1, Q, 1, 1)``, or ``exactness_degree`` is negative.
        """
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


# Symmetric Dunavant / Hammer-Marlowe-Stroud rules on the reference triangle
# conv{(0, 0), (1, 0), (0, 1)}, keyed by exactness degree. Points are in
# Cartesian reference coordinates and weights sum to one.
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
    """
    Smallest tabulated degree that reaches the requested exactness.

    Parameters
    ----------
    available_degrees : list[int]
        Tabulated exactness degrees, sorted in increasing order.
    exactness_degree : int
        Requested exactness degree, at least one.
    cell_name : str
        Reference cell name used in error messages.

    Returns
    -------
    selected_degree : int
        The smallest available degree that is at least ``exactness_degree``.
    """
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
    """
    Triangle quadrature rule exact to at least the requested degree.

    Parameters
    ----------
    reference_cell : ReferenceCell
        The reference triangle the rule is attached to.
    exactness_degree : int
        Minimum polynomial exactness degree, at least one.

    Returns
    -------
    quadrature : QuadratureRule
        Rule with points of shape ``(1, Q, 1, 2)`` and weights of shape
        ``(1, Q, 1, 1)``.

    Notes
    -----
    The rule is the smallest tabulated one reaching ``exactness_degree``; its
    ``exactness_degree`` attribute records the actual degree, which may be
    higher than requested.
    """
    selected_degree = _select_quadrature_degree(
        sorted(_TRIANGLE_RULES), exactness_degree, "triangle"
    )
    raw_points, raw_weights = _TRIANGLE_RULES[selected_degree]
    return QuadratureRule(
        reference_cell=reference_cell,
        points=jnp.expand_dims(jnp.asarray(raw_points), axis=(-4, -2)),
        weights=jnp.expand_dims(jnp.asarray(raw_weights), axis=(-4, -2, -1)),
        exactness_degree=selected_degree,
    )


# Gauss-Legendre rules on the reference interval [0, 1], keyed by exactness
# degree. An n-point rule is exact to degree 2n - 1; the standard [-1, 1]
# nodes and weights are mapped affinely to [0, 1], so the weights sum to one.
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
    """
    Interval quadrature rule exact to at least the requested degree.

    Parameters
    ----------
    reference_cell : ReferenceCell
        The reference interval the rule is attached to.
    exactness_degree : int
        Minimum polynomial exactness degree, at least one.

    Returns
    -------
    quadrature : QuadratureRule
        Rule with points of shape ``(1, Q, 1, 1)`` and weights of shape
        ``(1, Q, 1, 1)``.

    Notes
    -----
    Selection follows ``create_triangle_quadrature``.
    """
    selected_degree = _select_quadrature_degree(
        sorted(_INTERVAL_RULES), exactness_degree, "interval"
    )
    raw_points, raw_weights = _INTERVAL_RULES[selected_degree]
    return QuadratureRule(
        reference_cell=reference_cell,
        points=jnp.expand_dims(jnp.asarray(raw_points), axis=(-4, -2, -1)),
        weights=jnp.expand_dims(jnp.asarray(raw_weights), axis=(-4, -2, -1)),
        exactness_degree=selected_degree,
    )
