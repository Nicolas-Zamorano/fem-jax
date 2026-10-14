"""
Straight-sided triangular meshes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

# Local facet i is opposite local vertex i, as in
# ReferenceTriangle.facets_to_vertices.
_LOCAL_FACETS_TO_LOCAL_VERTICES = np.array([[1, 2], [2, 0], [0, 1]])

#: A mapping from boundary tag name to a predicate over boundary facet
#: midpoint coordinates, shape ``(N_boundary_facets, 1, d)``, returning a
#: boolean array of shape ``(N_boundary_facets,)``.
BoundaryData: TypeAlias = Mapping[str, Callable[[jax.Array], jax.Array]]

_DEFAULT_BOUNDARY_TAG_NAME = "boundary"


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class TriangleMesh:
    """
    Geometry, topology, and tags of a conforming triangular mesh.

    Attributes
    ----------
    vertex_coordinates : jax.Array
        Vertex coordinates as row vectors, shape ``(N_vertices, 1, 2)``.
    cells_to_vertices : jax.Array
        Vertex indices of each cell, shape ``(K, 3)``.
    facets_to_vertices : jax.Array
        Canonical ``(low, high)`` vertex pair of each facet, shape
        ``(N_facets, 2)``.
    cells_to_facets : jax.Array
        Global facet index of each local facet, shape ``(K, 3)``.
    facets_to_cells : jax.Array
        Cells adjacent to each facet, with ``-1`` for the missing neighbour
        of a boundary facet, shape ``(N_facets, 2)``.
    boundary_facets : jax.Array
        Facets with a single adjacent cell, shape ``(N_boundary_facets,)``.
    boundary_facet_tags : jax.Array
        Tag of each boundary facet, shape ``(N_boundary_facets,)``.
    cell_tags : jax.Array | None
        Physical tag of each cell, shape ``(K,)``, or ``None``.
    boundary_tag_names : Mapping[str, int]
        Mapping from boundary tag name to its integer value.
    vertices_to_cells : jax.Array
        Cells incident to each vertex (the vertex star), padded with ``0``,
        shape ``(N_vertices, max_valence)``.
    vertices_to_local_vertex : jax.Array
        Local index (0, 1, or 2) of the vertex within each incident cell,
        padded with ``0``, shape ``(N_vertices, max_valence)``.
    vertex_patch_mask : jax.Array
        ``True`` for real incident cells and ``False`` for padding, shape
        ``(N_vertices, max_valence)``.
    vertex_patch_diameters : jax.Array
        Diameter of each vertex star, the largest distance between two of
        its vertices, shape ``(N_vertices,)``.

    Notes
    -----
    The vertex-star arrays are padded rather than ragged. Padding uses index
    ``0`` because JAX gathers wrap a ``-1`` index silently; consumers must
    apply ``vertex_patch_mask``.
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
    vertices_to_cells: jax.Array
    vertices_to_local_vertex: jax.Array
    vertex_patch_mask: jax.Array
    vertex_patch_diameters: jax.Array

    @property
    def geometric_dimension(self) -> int:
        """
        Geometric dimension of the mesh.

        Returns
        -------
        geometric_dimension : int
            Always ``2``.
        """
        return 2

    @property
    def topological_dimension(self) -> int:
        """
        Topological dimension of the mesh.

        Returns
        -------
        topological_dimension : int
            Always ``2``.
        """
        return 2


def cell_diameters(mesh: TriangleMesh) -> jax.Array:
    """
    Diameter (longest edge length) of each cell.

    Parameters
    ----------
    mesh : TriangleMesh
        The mesh.

    Returns
    -------
    diameters : jax.Array
        Cell diameters, shape ``(K,)``.
    """
    vertices = mesh.vertex_coordinates[mesh.cells_to_vertices]
    edges = vertices - jnp.roll(vertices, shift=-1, axis=-3)
    edge_lengths = jnp.linalg.norm(edges, axis=-1)
    return jnp.max(edge_lengths, axis=(-2, -1))


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
    Build a triangular mesh from vertex and connectivity arrays.

    Parameters
    ----------
    vertex_coordinates : ArrayLike
        Vertex coordinates, shape ``(N_vertices, 2)``.
    cells_to_vertices : ArrayLike
        Vertex indices of each cell, shape ``(K, 3)``.
    boundary_data : BoundaryData | None
        Boundary tagging by predicates on boundary facet midpoints.
    boundary_facet_tags : Mapping[tuple[int, int], int] | None
        Boundary tagging by canonical ``(low, high)`` vertex pair; every
        boundary facet must be present. Requires ``boundary_tag_names``.
    boundary_tag_names : Mapping[str, int] | None
        Tag name to integer value, used with ``boundary_facet_tags``.
    cell_tags : ArrayLike | None
        Physical tag of each cell, shape ``(K,)``.

    Returns
    -------
    mesh : TriangleMesh
        The mesh with facet topology, boundary tags, and vertex stars.

    Notes
    -----
    Facet topology is derived from ``cells_to_vertices`` alone. Boundary tags
    come from at most one of ``boundary_data`` and ``boundary_facet_tags``;
    without either, every boundary facet gets the tag ``"boundary"`` (value
    ``0``). Explicit tags suit callers that already know each edge's tag,
    such as refinement propagating tags from a parent mesh. Cell orientation
    is not checked; cells are expected to be consistently oriented.
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

    (
        vertices_to_cells_np,
        vertices_to_local_vertex_np,
        vertex_patch_mask_np,
        vertex_patch_diameters_np,
    ) = _build_vertex_stars(vertex_coordinates_np, cells_to_vertices_np)

    return TriangleMesh(
        vertex_coordinates=jnp.expand_dims(jnp.asarray(vertex_coordinates_np), axis=-2),
        cells_to_vertices=jnp.asarray(cells_to_vertices_np, dtype=jnp.int32),
        facets_to_vertices=jnp.asarray(facets_to_vertices_np, dtype=jnp.int32),
        cells_to_facets=jnp.asarray(cells_to_facets_np, dtype=jnp.int32),
        facets_to_cells=jnp.asarray(facets_to_cells_np, dtype=jnp.int32),
        boundary_facets=jnp.asarray(boundary_facets_np, dtype=jnp.int32),
        boundary_facet_tags=jnp.asarray(boundary_facet_tags_np, dtype=jnp.int32),
        cell_tags=cell_tags_jnp,
        boundary_tag_names=resolved_boundary_tag_names,
        vertices_to_cells=jnp.asarray(vertices_to_cells_np, dtype=jnp.int32),
        vertices_to_local_vertex=jnp.asarray(
            vertices_to_local_vertex_np, dtype=jnp.int32
        ),
        vertex_patch_mask=jnp.asarray(vertex_patch_mask_np),
        vertex_patch_diameters=jnp.asarray(vertex_patch_diameters_np),
    )


def _build_vertex_stars(
    vertex_coordinates: np.ndarray,
    cells_to_vertices: np.ndarray,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Padded incidence and diameters of every vertex star.

    Parameters
    ----------
    vertex_coordinates : np.ndarray
        Vertex coordinates, shape ``(N_vertices, 2)``.
    cells_to_vertices : np.ndarray
        Vertex indices of every cell, shape ``(K, 3)``.
    chunk_size : int
        Number of vertices whose diameters are computed at once.

    Returns
    -------
    vertices_to_cells : np.ndarray
        Incident cells, padded with ``0``, shape ``(N_vertices, max_valence)``.
    vertices_to_local_vertex : np.ndarray
        Local vertex index in each incident cell, padded with ``0``, shape
        ``(N_vertices, max_valence)``.
    vertex_patch_mask : np.ndarray
        ``True`` for real incidences, shape ``(N_vertices, max_valence)``.
    vertex_patch_diameters : np.ndarray
        Largest distance between two vertices of each star, shape
        ``(N_vertices,)``; ``0`` for a vertex in no cell.

    Notes
    -----
    Incidences are grouped by a stable sort of the flattened
    ``cells_to_vertices``, so cells appear in increasing order within each
    star. For the diameters, padded slots are replaced by the star's own
    centre vertex, which cannot change the maximum distance.
    """
    number_of_vertices = vertex_coordinates.shape[0]
    number_of_cells = cells_to_vertices.shape[0]

    flat_vertices = np.ravel(cells_to_vertices)
    flat_cells = np.repeat(np.arange(number_of_cells), 3)
    flat_local_vertices = np.tile(np.arange(3), number_of_cells)

    order = np.argsort(flat_vertices, kind="stable")
    sorted_vertices = flat_vertices[order]
    valences = np.bincount(flat_vertices, minlength=number_of_vertices)
    max_valence = int(valences.max())
    first_slot = np.cumsum(valences) - valences
    slots = np.arange(flat_vertices.shape[0]) - first_slot[sorted_vertices]

    shape = (number_of_vertices, max_valence)
    vertices_to_cells = np.zeros(shape, dtype=np.int64)
    vertices_to_local_vertex = np.zeros(shape, dtype=np.int64)
    vertex_patch_mask = np.zeros(shape, dtype=bool)
    vertices_to_cells[sorted_vertices, slots] = flat_cells[order]
    vertices_to_local_vertex[sorted_vertices, slots] = flat_local_vertices[order]
    vertex_patch_mask[sorted_vertices, slots] = True

    star_vertices = np.where(
        np.expand_dims(vertex_patch_mask, axis=-1),
        cells_to_vertices[vertices_to_cells],
        np.expand_dims(np.arange(number_of_vertices), axis=(-2, -1)),
    ).reshape(number_of_vertices, 3 * max_valence)

    vertex_patch_diameters = np.zeros(number_of_vertices)
    for start in range(0, number_of_vertices, chunk_size):
        points = vertex_coordinates[star_vertices[start : start + chunk_size]]
        differences = np.expand_dims(points, axis=-2) - np.expand_dims(points, axis=-3)
        distances = np.linalg.norm(differences, axis=-1)
        vertex_patch_diameters[start : start + chunk_size] = np.max(
            distances, axis=(-2, -1)
        )

    return (
        vertices_to_cells,
        vertices_to_local_vertex,
        vertex_patch_mask,
        vertex_patch_diameters,
    )


def _validate_array_mesh_input(
    vertex_coordinates: np.ndarray, cells_to_vertices: np.ndarray
) -> None:
    """
    Check the shapes and index range of raw mesh arrays.

    Parameters
    ----------
    vertex_coordinates : np.ndarray
        Vertex coordinates, expected shape ``(N_vertices, 2)``.
    cells_to_vertices : np.ndarray
        Vertex indices of each cell, expected shape ``(K, 3)``.

    Raises
    ------
    ValueError
        If a shape is wrong, there are no cells, or an index is out of range.
    """
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
    """
    Facet connectivity derived from the cell vertices.

    Parameters
    ----------
    cells_to_vertices : np.ndarray
        Vertex indices of each cell, shape ``(K, 3)``.

    Returns
    -------
    facets_to_vertices : np.ndarray
        Canonical ``(low, high)`` vertex pair of each facet, shape
        ``(N_facets, 2)``.
    cells_to_facets : np.ndarray
        Global facet index of each local facet, shape ``(K, 3)``.
    facets_to_cells : np.ndarray
        Adjacent cells, ``-1`` for a missing second cell, shape
        ``(N_facets, 2)``.
    boundary_facets : np.ndarray
        Facets with a single adjacent cell, shape ``(N_boundary_facets,)``.

    Raises
    ------
    ValueError
        If a facet is shared by more than two cells.
    """
    number_of_cells = cells_to_vertices.shape[0]

    local_facet_vertices = cells_to_vertices[:, _LOCAL_FACETS_TO_LOCAL_VERTICES]
    flat_facet_vertices = local_facet_vertices.reshape(3 * number_of_cells, 2)
    canonical_facet_vertices = np.sort(flat_facet_vertices, axis=-1)

    facets_to_vertices, inverse = np.unique(
        canonical_facet_vertices, axis=0, return_inverse=True
    )
    inverse = np.ravel(inverse)
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
    """
    Boundary facet tags from an explicit vertex-pair map.

    Parameters
    ----------
    facets_to_vertices : np.ndarray
        Canonical vertex pair of each facet, shape ``(N_facets, 2)``.
    boundary_facets : np.ndarray
        Boundary facet indices, shape ``(N_boundary_facets,)``.
    boundary_facet_tags : Mapping[tuple[int, int], int]
        Tag of each boundary facet, keyed by canonical vertex pair.
    boundary_tag_names : Mapping[str, int]
        Tag name to integer value.

    Returns
    -------
    tags : np.ndarray
        Tag of each boundary facet, shape ``(N_boundary_facets,)``.
    boundary_tag_names : dict[str, int]
        Copy of ``boundary_tag_names``.
    """
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
    """
    Boundary facet tags from predicates on facet midpoints.

    Parameters
    ----------
    vertex_coordinates : np.ndarray
        Vertex coordinates, shape ``(N_vertices, d)``.
    facets_to_vertices : np.ndarray
        Canonical vertex pair of each facet, shape ``(N_facets, 2)``.
    boundary_facets : np.ndarray
        Boundary facet indices, shape ``(N_boundary_facets,)``.
    boundary_data : BoundaryData | None
        Tag name to predicate; ``None`` tags every facet ``"boundary"``.

    Returns
    -------
    tags : np.ndarray
        Tag of each boundary facet, shape ``(N_boundary_facets,)``.
    boundary_tag_names : dict[str, int]
        Tag name to integer value, numbered in predicate order.

    Notes
    -----
    A facet matched by several predicates keeps the first match.
    """
    if boundary_data is None:
        boundary_data = {
            _DEFAULT_BOUNDARY_TAG_NAME: lambda midpoints: jnp.ones(
                midpoints.shape[0], dtype=bool
            )
        }

    boundary_tag_names = {name: index for index, name in enumerate(boundary_data)}

    number_of_boundary_facets = boundary_facets.shape[0]
    boundary_facet_vertices = facets_to_vertices[boundary_facets]
    midpoints = vertex_coordinates[boundary_facet_vertices].mean(axis=-2)
    midpoints_for_predicates = jnp.expand_dims(jnp.asarray(midpoints), axis=-2)

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
