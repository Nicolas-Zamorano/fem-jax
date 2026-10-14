"""
The reference interval, used for facet quadrature.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem.reference_cell.quadrature import (
    QuadratureRule,
    create_interval_quadrature,
)

_VERTEX_COORDINATES = ((0.0,), (1.0,))

# The facets of a 1D cell are its two endpoints; they complete the
# ReferenceCell protocol but are not used by facet integration.
_FACETS_TO_VERTICES = ((0,), (1,))
_FACET_NORMALS = ((-1.0,), (1.0,))

_MEASURE = 1.0


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class ReferenceInterval:
    """
    The reference interval ``[0, 1]``.

    Attributes
    ----------
    dimension : int
        Always ``1``.
    vertex_coordinates : jax.Array
        Endpoint coordinates, shape ``(2, 1)``.
    facets_to_vertices : jax.Array
        Local vertex index of each endpoint, shape ``(2, 1)``.
    facet_normals : jax.Array
        Outward unit direction at each endpoint, shape ``(2, 1)``.
    measure : float
        Length of the interval, ``1.0``.

    Notes
    -----
    Every facet of a ``TriangleMesh`` is a straight segment parameterized by
    this interval, so it is the reference cell of ``FacetBasis`` quadrature.
    The class takes no constructor arguments.
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
        """
        Quadrature rule on this interval.

        Parameters
        ----------
        exactness_degree : int
            Minimum polynomial exactness degree, at least one.

        Returns
        -------
        quadrature : QuadratureRule
            Rule exact to at least ``exactness_degree`` (see
            ``create_interval_quadrature``).
        """
        return create_interval_quadrature(self, exactness_degree)
