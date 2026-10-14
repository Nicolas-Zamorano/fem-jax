"""
Reference-cell interface shared by all reference cell types.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax

from jax_fem.reference_cell.quadrature import QuadratureRule


@runtime_checkable
class ReferenceCell(Protocol):
    """
    Geometry and topology of a reference cell.

    Attributes
    ----------
    dimension : int
        Dimension of the reference cell.
    vertex_coordinates : jax.Array
        Vertex coordinates, shape ``(n_vertices, dimension)``.
    facets_to_vertices : jax.Array
        Local vertex indices of each facet, shape
        ``(n_facets, n_vertices_per_facet)``.
    facet_normals : jax.Array
        Outward unit normal of each facet, shape ``(n_facets, dimension)``.
    measure : float
        Exact measure (length, area, or volume) of the reference cell.

    Notes
    -----
    Physical cells of one topological type are images of their reference
    cell. ``measure`` converts normalized quadrature weights into physical
    weights. A reference cell holds no basis and no physical mesh data.
    """

    dimension: int
    vertex_coordinates: jax.Array
    facets_to_vertices: jax.Array
    facet_normals: jax.Array
    measure: float

    def create_quadrature(self, exactness_degree: int) -> QuadratureRule:
        """
        Quadrature rule on this cell.

        Parameters
        ----------
        exactness_degree : int
            Minimum polynomial exactness degree.

        Returns
        -------
        quadrature : QuadratureRule
            Rule exact for polynomials of total degree up to
            ``exactness_degree``; its actual exactness may be higher.
        """
        ...
