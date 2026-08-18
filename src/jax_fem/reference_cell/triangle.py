"""The canonical reference triangle.

See Section 8.2 of the architecture specification.
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

# The standard reference triangle K_hat = conv{(0, 0), (1, 0), (0, 1)}.
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
    """The standard reference triangle ``conv{(0, 0), (1, 0), (0, 1)}``.

    Stores only geometric and topological reference data and provides
    access to triangle-compatible quadrature rules. It owns no
    finite-element basis and no physical mesh (Section 20). All fields are
    fixed constants of the reference cell; the class takes no constructor
    arguments.

    Attributes
    ----------
    dimension:
        Always ``2``.
    vertex_coordinates:
        Reference vertex coordinates, shape ``(3, 2)``.
    facets_to_vertices:
        Local vertex indices of each reference edge, shape ``(3, 2)``.
        Facet ``i`` is opposite local vertex ``i``.
    facet_normals:
        Outward unit normal of each reference edge, shape ``(3, 2)``.
    measure:
        Exact area of the reference triangle, ``0.5``.
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
        """Return a triangle quadrature rule exact to at least ``exactness_degree``.

        See ``create_triangle_quadrature`` for the exactness-selection rule.
        """
        return create_triangle_quadrature(self, exactness_degree)
