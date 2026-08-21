"""The canonical reference interval, used for 1D facet quadrature.

See Section 4.2 of the implementation plan (``.context/implementation_plan.md``).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem.reference_cell.quadrature import (
    QuadratureRule,
    create_interval_quadrature,
)

# The standard reference interval [0, 1].
_VERTEX_COORDINATES = ((0.0,), (1.0,))

# The "facets" of a 1D cell are its two endpoints (0-dimensional points).
# Not exercised by facet-jump computations (only 2D triangle facets --
# edges -- are), but completes the ReferenceCell protocol.
_FACETS_TO_VERTICES = ((0,), (1,))
_FACET_NORMALS = ((-1.0,), (1.0,))

_MEASURE = 1.0


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class ReferenceInterval:
    """The standard reference interval ``[0, 1]``.

    Used as the reference cell for 1D facet quadrature (``FacetBasis``):
    every mesh facet of a ``TriangleMesh`` is a straight line segment, whose
    reference parameterization is this interval. Stores only geometric and
    topological reference data, matching ``ReferenceTriangle``'s role for
    2D cells (Section 8.2); the class takes no constructor arguments.

    Attributes
    ----------
    dimension:
        Always ``1``.
    vertex_coordinates:
        Reference endpoint coordinates, shape ``(2, 1)``.
    facets_to_vertices:
        Local vertex index of each reference endpoint, shape ``(2, 1)``.
    facet_normals:
        Outward unit "normal" (direction) of each reference endpoint, shape
        ``(2, 1)``.
    measure:
        Exact length of the reference interval, ``1.0``.
    """

    dimension: int = dataclasses.field(default=1, init=False)
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
        """Return an interval quadrature rule exact to at least ``exactness_degree``.

        See ``create_interval_quadrature`` for the exactness-selection rule.
        """
        return create_interval_quadrature(self, exactness_degree)
