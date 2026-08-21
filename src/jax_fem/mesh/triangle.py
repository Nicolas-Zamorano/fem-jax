"""Straight-sided triangular meshes.

See Sections 7.2 and 7.3 of the architecture specification.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

# Local facet i is opposite local vertex i, matching
# ReferenceTriangle.facets_to_vertices:
#   facet 0 = (v1, v2), facet 1 = (v2, v0), facet 2 = (v0, v1).
_LOCAL_FACETS_TO_LOCAL_VERTICES = np.array([[1, 2], [2, 0], [0, 1]])

#: A mapping from boundary tag name to a predicate over boundary facet
#: midpoint coordinates, shape ``(N_boundary_facets, 1, 2)``, returning a
#: boolean array of shape ``(N_boundary_facets,)``.
BoundaryData: TypeAlias = Mapping[str, Callable[[jax.Array], jax.Array]]

_DEFAULT_BOUNDARY_TAG_NAME = "boundary"


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class TriangleMesh:
    """
    Geometry, topology, and mesh labels of a conforming triangular mesh.

    The mesh owns all available physical and boundary tags.

    Attributes
    ----------
    vertex_coordinates : jax.Array
        Shape ``(N_vertices, 1, 2)``: each vertex is a row vector of shape
        ``(1, 2)``, matching the coefficient callable contract directly.
    cells_to_vertices : jax.Array
        Shape ``(K, 3)``.
    facets_to_vertices : jax.Array
        Shape ``(N_facets, 2)``, canonical ``(low, high)`` vertex index
        pairs.
    cells_to_facets : jax.Array
        Shape ``(K, 3)``, global facet index of each local facet.
    facets_to_cells : jax.Array
        Shape ``(N_facets, 2)``, adjacent cell indices; a missing second
        cell (a boundary facet) uses the sentinel ``-1``.
    boundary_facets : jax.Array
        Shape ``(N_boundary_facets,)``, facet indices with only one
        adjacent cell.
    boundary_facet_tags : jax.Array
        Shape ``(N_boundary_facets,)``, integer tag of each boundary facet.
    cell_tags : jax.Array | None
        Shape ``(K,)`` physical cell tags, or ``None`` if untagged.
    boundary_tag_names : Mapping[str, int]
        Mapping from boundary tag name to its integer value.
    """

    vertex_coordinates: jax.Array
    cells_to_vertices: jax.Array
    facets_to_vertices: jax.Array
    cells_to_facets: jax.Array
    facets_to_cells: jax.Array
    boundary_facets: jax.Array
    boundary_facet_tags: jax.Array
    cell_tags: jax.Array | None
    boundary_tag_names: Mapping[str, int]

    @property
    def geometric_dimension(self) -> int:
        """
        Geometric dimension of the mesh.

        Returns
        -------
        int
            Geometric dimension of the mesh.
        """
        return 2

    @property
    def topological_dimension(self) -> int:
        """
        Topological dimension of the mesh.

        Returns
        -------
        int
            Topological dimension of the mesh.
        """
        return 2


def cell_diameters(mesh: TriangleMesh) -> jax.Array:
    """
    Cell diameter (longest edge length) of each cell.

    Parameters
    ----------
    mesh : TriangleMesh

    Returns
    -------
    jax.Array
        Shape ``(K,)``.
    """
    vertices = mesh.vertex_coordinates[mesh.cells_to_vertices]  # (K, 3, 1, 2)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]  # each (K, 1, 2)
    edge_lengths = jnp.concatenate(
        (
            jnp.linalg.norm(v1 - v0, axis=-1),
            jnp.linalg.norm(v2 - v1, axis=-1),
            jnp.linalg.norm(v0 - v2, axis=-1),
        ),
        axis=-1,
    )  # (K, 3)
    return jnp.max(edge_lengths, axis=-1)


def create_triangle_mesh_from_arrays(
    vertex_coordinates: ArrayLike,
    cells_to_vertices: ArrayLike,
    *,
    boundary_data: BoundaryData | None = None,
    boundary_facet_tags: Mapping[tuple[int, int], int] | None = None,
    boundary_tag_names: Mapping[str, int] | None = None,
    cell_tags: ArrayLike | None = None,
) -> TriangleMesh:
    """
    Build a ``TriangleMesh`` from raw vertex and connectivity arrays.

    Facet topology (``facets_to_vertices``, ``cells_to_facets``,
    ``facets_to_cells``, ``boundary_facets``) is derived purely from
    ``cells_to_vertices``; the caller never enumerates facets by hand.
    Boundary tagging is supplied through exactly one of two mutually
    exclusive paths, ``boundary_data`` or ``boundary_facet_tags``.

    Cell orientation (clockwise vs. counter-clockwise) is not validated or
    corrected: the caller is trusted to supply consistently oriented cells.

    Parameters
    ----------
    vertex_coordinates : ArrayLike
        Array-like of shape ``(N_vertices, 2)``.
    cells_to_vertices : ArrayLike
        Array-like of shape ``(K, 3)`` of vertex indices into
        ``vertex_coordinates``.
    boundary_data : BoundaryData | None
        Predicate-based boundary tagging, evaluated on boundary facet
        midpoints. If neither ``boundary_data`` nor ``boundary_facet_tags``
        is given, every boundary facet receives one default tag (name
        ``"boundary"``, value ``0``). Mutually exclusive with
        ``boundary_facet_tags``.
    boundary_facet_tags : Mapping[tuple[int, int], int] | None
        Explicit boundary tagging, keyed by canonical ``(low, high)``
        vertex-index pair (mirroring ``facets_to_vertices``'s own
        convention) rather than by predicate -- the caller already knows
        which tag each boundary edge carries (e.g. a mesh refinement pass
        propagating tags from a parent mesh) and doesn't need coordinate
        matching to rediscover it. Every detected boundary facet's pair
        must be present. Requires ``boundary_tag_names``. Mutually
        exclusive with ``boundary_data``.
    boundary_tag_names : Mapping[str, int] | None
        Tag name -> integer value, paired with ``boundary_facet_tags``.
    cell_tags : ArrayLike | None
        Optional physical cell tags, array-like of shape ``(K,)``. If
        ``None``, ``TriangleMesh.cell_tags`` is ``None``.

    Returns
    -------
    TriangleMesh
    """
    if boundary_data is not None and boundary_facet_tags is not None:
        raise ValueError(
            "boundary_data and boundary_facet_tags are mutually exclusive; "
            "pass at most one."
        )
    if boundary_facet_tags is not None and boundary_tag_names is None:
        raise ValueError(
            "boundary_tag_names is required when boundary_facet_tags is given."
        )

    vertex_coordinates_np = np.asarray(vertex_coordinates, dtype=np.float64)
    cells_to_vertices_np = np.asarray(cells_to_vertices, dtype=np.int64)
    _validate_array_mesh_input(vertex_coordinates_np, cells_to_vertices_np)

    (
        facets_to_vertices_np,
        cells_to_facets_np,
        facets_to_cells_np,
        boundary_facets_np,
    ) = _build_facet_topology(cells_to_vertices_np)

    if boundary_facet_tags is not None:
        boundary_facet_tags_np, resolved_boundary_tag_names = (
            _tag_boundary_facets_explicitly(
                facets_to_vertices_np,
                boundary_facets_np,
                boundary_facet_tags,
                boundary_tag_names,
            )
        )
    else:
        boundary_facet_tags_np, resolved_boundary_tag_names = _tag_boundary_facets(
            vertex_coordinates_np,
            facets_to_vertices_np,
            boundary_facets_np,
            boundary_data,
        )

    cell_tags_jnp = None
    if cell_tags is not None:
        cell_tags_np = np.asarray(cell_tags, dtype=np.int64)
        if cell_tags_np.shape != (cells_to_vertices_np.shape[0],):
            raise ValueError(
                "cell_tags must have shape "
                f"({cells_to_vertices_np.shape[0]},), got {cell_tags_np.shape}."
            )
        cell_tags_jnp = jnp.asarray(cell_tags_np, dtype=jnp.int32)

    return TriangleMesh(
        vertex_coordinates=jnp.expand_dims(
            jnp.asarray(vertex_coordinates_np), axis =-2
            ),
        cells_to_vertices=jnp.asarray(cells_to_vertices_np, dtype=jnp.int32),
        facets_to_vertices=jnp.asarray(facets_to_vertices_np, dtype=jnp.int32),
        cells_to_facets=jnp.asarray(cells_to_facets_np, dtype=jnp.int32),
        facets_to_cells=jnp.asarray(facets_to_cells_np, dtype=jnp.int32),
        boundary_facets=jnp.asarray(boundary_facets_np, dtype=jnp.int32),
        boundary_facet_tags=jnp.asarray(boundary_facet_tags_np, dtype=jnp.int32),
        cell_tags=cell_tags_jnp,
        boundary_tag_names=resolved_boundary_tag_names,
    )


def _validate_array_mesh_input(
    vertex_coordinates: np.ndarray, cells_to_vertices: np.ndarray
) -> None:
    if vertex_coordinates.ndim != 2 or vertex_coordinates.shape[1] != 2:
        raise ValueError(
            "vertex_coordinates must have shape (N_vertices, 2), got "
            f"{vertex_coordinates.shape}."
        )
    if cells_to_vertices.ndim != 2 or cells_to_vertices.shape[1] != 3:
        raise ValueError(
            "cells_to_vertices must have shape (K, 3), got "
            f"{cells_to_vertices.shape}."
        )
    if cells_to_vertices.shape[0] == 0:
        raise ValueError("cells_to_vertices must contain at least one cell.")

    number_of_vertices = vertex_coordinates.shape[0]
    if cells_to_vertices.min() < 0 or cells_to_vertices.max() >= number_of_vertices:
        raise ValueError(
            "cells_to_vertices references vertex indices outside "
            f"[0, {number_of_vertices})."
        )


def _build_facet_topology(
    cells_to_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive facet connectivity purely from cell-to-vertex topology.

    Returns
    -------
    facets_to_vertices:
        ``(N_facets, 2)``, canonical ``(low, high)`` vertex index pairs.
    cells_to_facets:
        ``(K, 3)``, global facet index of each local facet.
    facets_to_cells:
        ``(N_facets, 2)``, adjacent cell indices, ``-1`` sentinel for a
        missing second cell.
    boundary_facets:
        ``(N_boundary_facets,)``, facet indices with only one adjacent cell.
    """
    number_of_cells = cells_to_vertices.shape[0]

    # (K, 3, 2): for each cell and local facet, its two global vertex ids.
    local_facet_vertices = cells_to_vertices[:, _LOCAL_FACETS_TO_LOCAL_VERTICES]
    flat_facet_vertices = local_facet_vertices.reshape(-1, 2)
    canonical_facet_vertices = np.sort(flat_facet_vertices, axis=1)

    facets_to_vertices, inverse = np.unique(
        canonical_facet_vertices, axis=0, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    number_of_facets = facets_to_vertices.shape[0]
    cells_to_facets = inverse.reshape(number_of_cells, 3)

    facet_occurrence_counts = np.bincount(inverse, minlength=number_of_facets)
    if np.any(facet_occurrence_counts > 2):
        bad_facets = np.nonzero(facet_occurrence_counts > 2)[0]
        raise ValueError(
            "Non-manifold mesh: facet(s) with vertices "
            f"{facets_to_vertices[bad_facets].tolist()} are shared by more "
            "than two cells."
        )

    cell_ids_flat = np.repeat(np.arange(number_of_cells), 3)
    order = np.argsort(inverse, kind="stable")
    sorted_facet_ids = inverse[order]
    sorted_cell_ids = cell_ids_flat[order]

    is_first_in_group = np.empty(sorted_facet_ids.shape[0], dtype=bool)
    is_first_in_group[0] = True
    is_first_in_group[1:] = sorted_facet_ids[1:] != sorted_facet_ids[:-1]

    facets_to_cells = np.full((number_of_facets, 2), -1, dtype=np.int64)
    facets_to_cells[sorted_facet_ids[is_first_in_group], 0] = sorted_cell_ids[
        is_first_in_group
    ]
    facets_to_cells[sorted_facet_ids[~is_first_in_group], 1] = sorted_cell_ids[
        ~is_first_in_group
    ]

    boundary_facets = np.nonzero(facets_to_cells[:, 1] == -1)[0]

    return facets_to_vertices, cells_to_facets, facets_to_cells, boundary_facets


def _tag_boundary_facets_explicitly(
    facets_to_vertices: np.ndarray,
    boundary_facets: np.ndarray,
    boundary_facet_tags: Mapping[tuple[int, int], int],
    boundary_tag_names: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign tags to already-detected boundary facets from an explicit map."""
    tags = np.empty(boundary_facets.shape[0], dtype=np.int64)
    for row, facet_index in enumerate(boundary_facets):
        vertex_a, vertex_b = (int(v) for v in facets_to_vertices[facet_index])
        key = (min(vertex_a, vertex_b), max(vertex_a, vertex_b))
        if key not in boundary_facet_tags:
            raise ValueError(
                f"Boundary facet with vertices {key} is not present in "
                "boundary_facet_tags."
            )
        tags[row] = boundary_facet_tags[key]
    return tags, dict(boundary_tag_names)


def _tag_boundary_facets(
    vertex_coordinates: np.ndarray,
    facets_to_vertices: np.ndarray,
    boundary_facets: np.ndarray,
    boundary_data: BoundaryData | None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign integer tags to already-detected boundary facets."""
    if boundary_data is None:
        boundary_data = {
            _DEFAULT_BOUNDARY_TAG_NAME: lambda midpoints: jnp.ones(
                midpoints.shape[0], dtype=bool
            )
        }

    boundary_tag_names = {name: index for index, name in enumerate(boundary_data)}

    number_of_boundary_facets = boundary_facets.shape[0]
    boundary_facet_vertices = facets_to_vertices[boundary_facets]  # (Nb, 2)
    midpoints = vertex_coordinates[boundary_facet_vertices].mean(axis=1)  # (Nb, 2)
    midpoints_for_predicates = jnp.asarray(midpoints).reshape(-1, 1, 2)

    tags = np.full(number_of_boundary_facets, -1, dtype=np.int64)
    for name, predicate in boundary_data.items():
        matched = np.asarray(predicate(midpoints_for_predicates))
        if matched.shape != (number_of_boundary_facets,):
            raise ValueError(
                f"Boundary predicate {name!r} must return a boolean array "
                f"of shape ({number_of_boundary_facets},), got "
                f"{matched.shape}."
            )
        newly_matched = matched & (tags == -1)
        tags[newly_matched] = boundary_tag_names[name]

    if np.any(tags == -1):
        unmatched = boundary_facets[tags == -1]
        raise ValueError(
            f"Boundary facets {unmatched.tolist()} matched no predicate in "
            "boundary_data."
        )

    return tags, boundary_tag_names
