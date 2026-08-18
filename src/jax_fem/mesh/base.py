"""
Mesh interface shared by all mesh types.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class Mesh(Protocol):
    """
    Geometry and topology of a physical mesh.

    A ``Mesh`` owns mesh geometry, topology, and physical/boundary tags.

    Attributes
    ----------
    geometric_dimension : int
        Dimension of the ambient space the vertex coordinates live in.
    topological_dimension : int
        Dimension of the mesh's cells.
    vertex_coordinates : jax.Array
        Coordinates of every mesh vertex, shape
        ``(N_vertices, 1, geometric_dimension)``.
    cells_to_vertices : jax.Array
        Vertex indices of every cell, shape ``(K, n_vertices_per_cell)``.
    """

    geometric_dimension: int
    topological_dimension: int
    vertex_coordinates: jax.Array
    cells_to_vertices: jax.Array
