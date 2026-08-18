"""Reference-cell interface shared by all reference cell types.

See Section 8.1 of the architecture specification.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax

from jax_fem.reference_cell.quadrature import QuadratureRule


@runtime_checkable
class ReferenceCell(Protocol):
    """Canonical geometry and topology of a reference cell.

    A ``ReferenceCell`` describes the fixed geometric object that all
    physical cells of one topological type are mapped from. It owns
    reference geometry, topology, and valid quadrature construction. It owns
    neither a finite-element basis nor any physical mesh data (Section 20).

    Attributes
    ----------
    dimension:
        Topological and geometric dimension of the reference cell.
    vertex_coordinates:
        Coordinates of the reference cell's vertices, shape
        ``(n_vertices, dimension)``.
    facets_to_vertices:
        Local vertex indices bounding each reference facet, shape
        ``(n_facets, n_vertices_per_facet)``.
    facet_normals:
        Outward unit normal of each reference facet, shape
        ``(n_facets, dimension)``.
    measure:
        Exact geometric measure (length/area/volume) of the reference cell
        itself. Not a quadrature-derived quantity: it is used only to
        convert normalized quadrature weights into physical integration
        weights (Section 6.3).
    """

    dimension: int
    vertex_coordinates: jax.Array
    facets_to_vertices: jax.Array
    facet_normals: jax.Array
    measure: float

    def create_quadrature(self, exactness_degree: int) -> QuadratureRule:
        """Return a quadrature rule valid on this cell.

        Parameters
        ----------
        exactness_degree:
            Minimum polynomial exactness degree required.

        Returns
        -------
        QuadratureRule
            A rule exact for polynomials of total degree at most
            ``exactness_degree`` (the actual exactness may be higher; see
            ``QuadratureRule.exactness_degree``).
        """
        ...
