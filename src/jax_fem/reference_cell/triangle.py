"""
The reference triangle.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from jax_fem.reference_cell.quadrature import (
    QuadratureRule,
    create_triangle_quadrature,
)

_VERTEX_COORDINATES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

# Local facet i is opposite local vertex i:
#   facet 0 = (v1, v2)  the hypotenuse
#   facet 1 = (v2, v0)  the left edge, x = 0
#   facet 2 = (v0, v1)  the bottom edge, y = 0
_FACETS_TO_VERTICES = ((1, 2), (2, 0), (0, 1))

_INVERSE_SQRT_2 = 1.0 / math.sqrt(2.0)
_FACET_NORMALS = ((_INVERSE_SQRT_2, _INVERSE_SQRT_2), (-1.0, 0.0), (0.0, -1.0))

_MEASURE = 0.5


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class ReferenceTriangle:
    """
    The reference triangle ``conv{(0, 0), (1, 0), (0, 1)}``.

    Attributes
    ----------
    dimension : int
        Always ``2``.
    vertex_coordinates : jax.Array
        Vertex coordinates, shape ``(3, 2)``.
    facets_to_vertices : jax.Array
        Local vertex indices of each edge, shape ``(3, 2)``; facet ``i`` is
        opposite local vertex ``i``.
    facet_normals : jax.Array
        Outward unit normal of each edge, shape ``(3, 2)``.
    measure : float
        Area of the triangle, ``0.5``.

    Notes
    -----
    Holds only reference geometry and topology, with no basis and no
    physical mesh. The class takes no constructor arguments.
    """

    dimension: int = dataclasses.field(default=2, init=False)
    vertex_coordinates: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(_VERTEX_COORDINATES), init=False
    )
    facets_to_vertices: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(_FACETS_TO_VERTICES, dtype=jnp.int32),
        init=False,
    )
    facet_normals: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(_FACET_NORMALS), init=False
    )
    measure: float = dataclasses.field(default=_MEASURE, init=False)

    def create_quadrature(self, exactness_degree: int) -> QuadratureRule:
        """
        Quadrature rule on this triangle.

        Parameters
        ----------
        exactness_degree : int
            Minimum polynomial exactness degree, at least one.

        Returns
        -------
        quadrature : QuadratureRule
            Rule exact to at least ``exactness_degree`` (see
            ``create_triangle_quadrature``).
        """
        return create_triangle_quadrature(self, exactness_degree)
