"""
Planar triangular meshes of discrete fracture networks embedded in 3D.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from jax_fem.mesh.triangle import (
    _LOCAL_FACETS_TO_LOCAL_VERTICES,
    BoundaryData,
    _tag_boundary_facets,
    _tag_boundary_facets_explicitly,
)

_MAX_FRACTURES_PER_FACET = 2
_MAX_CELLS_PER_FRACTURE_AND_FACET = 2
_MAX_CELLS_PER_FACET = _MAX_FRACTURES_PER_FACET * _MAX_CELLS_PER_FRACTURE_AND_FACET
_DEGENERACY_TOLERANCE = 1e-12


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class DfnMesh:
    """
    Geometry, topology, and tags of a discrete fracture network mesh.

    Attributes
    ----------
    vertex_coordinates : jax.Array
        Global vertex coordinates as row vectors, shape ``(N_vertices, 1, 3)``.
    cells_to_vertices : jax.Array
        Vertex indices of each cell, shape ``(K, 3)``.
    cell_tags : jax.Array
        Fracture tag of each cell, shape ``(K,)``.
    fracture_tags : jax.Array
        Distinct fracture tags in increasing order, shape ``(N_fractures,)``.
    cells_to_fractures : jax.Array
        Position of each cell's tag in ``fracture_tags``, shape ``(K,)``.
    fracture_normals : jax.Array
        Unit normal of each fracture plane, shape ``(N_fractures, 1, 3)``.
    facets_to_vertices : jax.Array
        Canonical ``(low, high)`` vertex pair of each facet, shape
        ``(N_facets, 2)``.
    cells_to_facets : jax.Array
        Global facet index of each local facet, shape ``(K, 3)``.
    facets_to_cells : jax.Array
        Cells incident to each facet in increasing order, padded with ``0``,
        shape ``(N_facets, 4)``.
    facet_cell_mask : jax.Array
        ``True`` for real incidences and ``False`` for padding, shape
        ``(N_facets, 4)``.
    boundary_facets : jax.Array
        Facets with a single incident cell, shape ``(N_boundary_facets,)``.
    boundary_facet_tags : jax.Array
        Tag of each boundary facet, shape ``(N_boundary_facets,)``.
    boundary_tag_names : Mapping[str, int]
        Mapping from boundary tag name to its integer value.
    trace_facets : jax.Array
        Facets whose incident cells belong to two fractures, grouped by trace
        and ordered along each trace, shape ``(N_trace_facets,)``.
    trace_facets_to_traces : jax.Array
        Trace of each entry of ``trace_facets``, non-decreasing, shape
        ``(N_trace_facets,)``.
    traces_to_fractures : jax.Array
        Positions in ``fracture_tags`` of the two fractures meeting at each
        trace, in increasing order, shape ``(N_traces, 2)``.
    trace_origins : jax.Array
        Start point of each trace, shape ``(N_traces, 1, 3)``.
    trace_directions : jax.Array
        Unit direction of each trace, shape ``(N_traces, 1, 3)``.

    Notes
    -----
    A trace facet has two incident cells when two fractures meet along
    their edges, three when a fracture ends on another, and four when two
    fractures cross. Any other facet has one incident cell (boundary) or two
    of the same fracture (interior). Trace vertices are shared by the
    fractures meeting there, so a conforming space is continuous across
    traces, and the Galerkin form balances the trace fluxes as its natural
    condition.

    ``vertex_coordinates``, ``cells_to_vertices``, ``facets_to_vertices``,
    ``cells_to_facets``, and the boundary fields mean the same as on
    ``TriangleMesh``. ``facets_to_cells`` differs: it has room for four
    cells and is padded with index ``0``, because JAX gathers wrap a ``-1``
    index silently; consumers must apply ``facet_cell_mask``.

    A trace is a connected set of trace facets shared by the same two
    fractures. Being part of the line where two fracture planes meet, it is
    straight, and ``(x - trace_origins) @ trace_directions.mT`` is the arc
    length of a point ``x`` on it, from ``0`` at the origin to the trace
    length at its end.

    Each fracture normal and trace direction is oriented so that its
    largest-magnitude component is positive.
    """

    vertex_coordinates: jax.Array
    cells_to_vertices: jax.Array
    cell_tags: jax.Array
    fracture_tags: jax.Array
    cells_to_fractures: jax.Array
    fracture_normals: jax.Array
    facets_to_vertices: jax.Array
    cells_to_facets: jax.Array
    facets_to_cells: jax.Array
    facet_cell_mask: jax.Array
    boundary_facets: jax.Array
    boundary_facet_tags: jax.Array
    boundary_tag_names: Mapping[str, int]
    trace_facets: jax.Array
    trace_facets_to_traces: jax.Array
    traces_to_fractures: jax.Array
    trace_origins: jax.Array
    trace_directions: jax.Array

    @property
    def geometric_dimension(self) -> int:
        """
        Geometric dimension of the mesh.

        Returns
        -------
        geometric_dimension : int
            Always ``3``.
        """
        return 3

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


def create_dfn_mesh_from_arrays(
    vertex_coordinates: ArrayLike,
    cells_to_vertices: ArrayLike,
    cell_tags: ArrayLike,
    *,
    boundary_data: BoundaryData | None = None,
    boundary_facet_tags: Mapping[tuple[int, int], int] | None = None,
    boundary_tag_names: Mapping[str, int] | None = None,
    coplanarity_tolerance: float = 1e-8,
) -> DfnMesh:
    """
    Build a fracture network mesh from vertex, connectivity, and tag arrays.

    Parameters
    ----------
    vertex_coordinates : ArrayLike
        Global vertex coordinates, shape ``(N_vertices, 3)``.
    cells_to_vertices : ArrayLike
        Vertex indices of each cell, shape ``(K, 3)``, in any winding.
    cell_tags : ArrayLike
        Fracture tag of each cell, shape ``(K,)``.
    boundary_data : BoundaryData | None
        Boundary tagging by predicates on boundary facet midpoints, shape
        ``(N_boundary_facets, 1, 3)``.
    boundary_facet_tags : Mapping[tuple[int, int], int] | None
        Boundary tagging by canonical ``(low, high)`` vertex pair; every
        boundary facet must be present. Requires ``boundary_tag_names``.
    boundary_tag_names : Mapping[str, int] | None
        Tag name to integer value, used with ``boundary_facet_tags``.
    coplanarity_tolerance : float
        Largest accepted distance from a fracture's vertices to its plane,
        relative to the diameter of the mesh's bounding box.

    Returns
    -------
    mesh : DfnMesh
        The mesh with facet incidence, traces, fracture normals, and
        boundary tags.

    Raises
    ------
    ValueError
        If an array has the wrong shape, an index is out of range, two
        vertices coincide, a cell is degenerate, a fracture is not planar, a
        facet is shared by more than two fractures, or a fracture has more
        than two cells on one facet.

    Notes
    -----
    Facets are identified by their vertex pair, so fractures are coupled
    only where they share vertices; coincident vertices are rejected
    because they would silently decouple a trace. Each fracture's plane is
    the least-squares plane through its vertices.

    Boundary tags come from at most one of ``boundary_data`` and
    ``boundary_facet_tags``; without either, every boundary facet gets the
    tag ``"boundary"`` (value ``0``). Boundary facets include the outer
    edges of the network and the edges of fractures that end inside
    another fracture's plane without a trace.
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
    cell_tags_np = np.asarray(cell_tags, dtype=np.int64)
    _validate_dfn_input(vertex_coordinates_np, cells_to_vertices_np, cell_tags_np)

    fracture_tags_np, cells_to_fractures_np = np.unique(
        cell_tags_np, return_inverse=True
    )
    cells_to_fractures_np = np.ravel(cells_to_fractures_np)

    _check_cells_are_non_degenerate(vertex_coordinates_np, cells_to_vertices_np)
    fracture_normals_np = _compute_fracture_normals(
        vertex_coordinates_np,
        cells_to_vertices_np,
        cells_to_fractures_np,
        coplanarity_tolerance,
    )

    (
        facets_to_vertices_np,
        cells_to_facets_np,
        facets_to_cells_np,
        facet_cell_mask_np,
        trace_facets_np,
        trace_fracture_pairs_np,
    ) = _build_dfn_facet_topology(cells_to_vertices_np, cells_to_fractures_np)
    (
        trace_facets_np,
        trace_facets_to_traces_np,
        traces_to_fractures_np,
        trace_origins_np,
        trace_directions_np,
    ) = _group_trace_facets(
        vertex_coordinates_np,
        facets_to_vertices_np,
        trace_facets_np,
        trace_fracture_pairs_np,
    )
    boundary_facets_np = np.flatnonzero(np.sum(facet_cell_mask_np, axis=-1) == 1)

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

    return DfnMesh(
        vertex_coordinates=jnp.expand_dims(jnp.asarray(vertex_coordinates_np), axis=-2),
        cells_to_vertices=jnp.asarray(cells_to_vertices_np, dtype=jnp.int32),
        cell_tags=jnp.asarray(cell_tags_np, dtype=jnp.int32),
        fracture_tags=jnp.asarray(fracture_tags_np, dtype=jnp.int32),
        cells_to_fractures=jnp.asarray(cells_to_fractures_np, dtype=jnp.int32),
        fracture_normals=jnp.expand_dims(jnp.asarray(fracture_normals_np), axis=-2),
        facets_to_vertices=jnp.asarray(facets_to_vertices_np, dtype=jnp.int32),
        cells_to_facets=jnp.asarray(cells_to_facets_np, dtype=jnp.int32),
        facets_to_cells=jnp.asarray(facets_to_cells_np, dtype=jnp.int32),
        facet_cell_mask=jnp.asarray(facet_cell_mask_np),
        boundary_facets=jnp.asarray(boundary_facets_np, dtype=jnp.int32),
        boundary_facet_tags=jnp.asarray(boundary_facet_tags_np, dtype=jnp.int32),
        boundary_tag_names=resolved_boundary_tag_names,
        trace_facets=jnp.asarray(trace_facets_np, dtype=jnp.int32),
        trace_facets_to_traces=jnp.asarray(trace_facets_to_traces_np, dtype=jnp.int32),
        traces_to_fractures=jnp.asarray(traces_to_fractures_np, dtype=jnp.int32),
        trace_origins=jnp.expand_dims(jnp.asarray(trace_origins_np), axis=-2),
        trace_directions=jnp.expand_dims(jnp.asarray(trace_directions_np), axis=-2),
    )


def _validate_dfn_input(
    vertex_coordinates: np.ndarray,
    cells_to_vertices: np.ndarray,
    cell_tags: np.ndarray,
) -> None:
    """
    Check the shapes, index range, and vertex uniqueness of raw DFN arrays.

    Parameters
    ----------
    vertex_coordinates : np.ndarray
        Vertex coordinates, expected shape ``(N_vertices, 3)``.
    cells_to_vertices : np.ndarray
        Vertex indices of each cell, expected shape ``(K, 3)``.
    cell_tags : np.ndarray
        Fracture tag of each cell, expected shape ``(K,)``.

    Raises
    ------
    ValueError
        If a shape is wrong, there are no cells, an index is out of range,
        or two vertices coincide.
    """
    if vertex_coordinates.ndim != 2 or vertex_coordinates.shape[-1] != 3:
        raise ValueError(
            "vertex_coordinates must have shape (N_vertices, 3), got "
            f"{vertex_coordinates.shape}."
        )
    if cells_to_vertices.ndim != 2 or cells_to_vertices.shape[-1] != 3:
        raise ValueError(
            f"cells_to_vertices must have shape (K, 3), got {cells_to_vertices.shape}."
        )
    if cells_to_vertices.shape[0] == 0:
        raise ValueError("cells_to_vertices must contain at least one cell.")
    if cell_tags.shape != (cells_to_vertices.shape[0],):
        raise ValueError(
            f"cell_tags must have shape ({cells_to_vertices.shape[0]},), got "
            f"{cell_tags.shape}."
        )

    number_of_vertices = vertex_coordinates.shape[0]
    if cells_to_vertices.min() < 0 or cells_to_vertices.max() >= number_of_vertices:
        raise ValueError(
            "cells_to_vertices references vertex indices outside "
            f"[0, {number_of_vertices})."
        )

    _, first_occurrences, multiplicities = np.unique(
        vertex_coordinates, axis=0, return_index=True, return_counts=True
    )
    if np.any(multiplicities > 1):
        duplicated = np.sort(first_occurrences[multiplicities > 1])
        raise ValueError(
            f"Vertices {duplicated.tolist()} have coincident copies; trace "
            "vertices must be shared by the fractures meeting there."
        )


def _check_cells_are_non_degenerate(
    vertex_coordinates: np.ndarray, cells_to_vertices: np.ndarray
) -> None:
    """
    Reject cells whose area vanishes relative to their longest edge.

    Parameters
    ----------
    vertex_coordinates : np.ndarray
        Vertex coordinates, shape ``(N_vertices, 3)``.
    cells_to_vertices : np.ndarray
        Vertex indices of each cell, shape ``(K, 3)``.

    Raises
    ------
    ValueError
        If ``2 area <= tolerance * longest_edge^2`` for some cell.
    """
    corners = vertex_coordinates[cells_to_vertices]
    edges = np.roll(corners, shift=-1, axis=-2) - corners
    doubled_areas = np.linalg.norm(
        np.cross(edges[..., 0, :], edges[..., 1, :]), axis=-1
    )
    longest_edges = np.max(np.linalg.norm(edges, axis=-1), axis=-1)
    degenerate = doubled_areas <= _DEGENERACY_TOLERANCE * longest_edges**2
    if np.any(degenerate):
        raise ValueError(f"Cells {np.flatnonzero(degenerate).tolist()} are degenerate.")


def _compute_fracture_normals(
    vertex_coordinates: np.ndarray,
    cells_to_vertices: np.ndarray,
    cells_to_fractures: np.ndarray,
    coplanarity_tolerance: float,
) -> np.ndarray:
    """
    Unit normal of each fracture's least-squares plane.

    Parameters
    ----------
    vertex_coordinates : np.ndarray
        Vertex coordinates, shape ``(N_vertices, 3)``.
    cells_to_vertices : np.ndarray
        Vertex indices of each cell, shape ``(K, 3)``.
    cells_to_fractures : np.ndarray
        Fracture index of each cell, shape ``(K,)``.
    coplanarity_tolerance : float
        Largest accepted distance to the plane, relative to the diameter of
        the bounding box of all vertices.

    Returns
    -------
    fracture_normals : np.ndarray
        Unit normals, oriented so that the largest-magnitude component is
        positive, shape ``(N_fractures, 3)``.

    Raises
    ------
    ValueError
        If a fracture's vertices are farther from its plane than the
        tolerance.

    Notes
    -----
    The normal is the right singular vector of the smallest singular value
    of the fracture's centred vertex coordinates.
    """
    number_of_fractures = int(cells_to_fractures.max()) + 1
    length_scale = np.linalg.norm(np.ptp(vertex_coordinates, axis=0))
    order = np.argsort(cells_to_fractures, kind="stable")
    boundaries = np.cumsum(np.bincount(cells_to_fractures))[:-1]
    cells_per_fracture = np.split(order, boundaries)

    fracture_normals = np.empty((number_of_fractures, 3))
    for fracture, cells in enumerate(cells_per_fracture):
        points = vertex_coordinates[np.unique(cells_to_vertices[cells])]
        centred = points - np.mean(points, axis=0)
        _, _, right_singular_vectors = np.linalg.svd(centred, full_matrices=False)
        normal = right_singular_vectors[-1]
        distance = np.max(np.abs(centred @ normal))
        if distance > coplanarity_tolerance * length_scale:
            raise ValueError(
                f"Fracture {fracture} is not planar: a vertex lies {distance:.3e} "
                "from its least-squares plane."
            )
        fracture_normals[fracture] = _orient_by_largest_component(normal)
    return fracture_normals


def _orient_by_largest_component(vector: np.ndarray) -> np.ndarray:
    """
    Flip a vector so that its largest-magnitude component is positive.

    Parameters
    ----------
    vector : np.ndarray
        The vector, shape ``(3,)``.

    Returns
    -------
    oriented : np.ndarray
        ``vector`` or ``-vector``, shape ``(3,)``.
    """
    return vector * np.sign(vector[np.argmax(np.abs(vector))])


def _build_dfn_facet_topology(
    cells_to_vertices: np.ndarray, cells_to_fractures: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-manifold facet connectivity derived from the cell vertices.

    Parameters
    ----------
    cells_to_vertices : np.ndarray
        Vertex indices of each cell, shape ``(K, 3)``.
    cells_to_fractures : np.ndarray
        Fracture index of each cell, shape ``(K,)``.

    Returns
    -------
    facets_to_vertices : np.ndarray
        Canonical ``(low, high)`` vertex pair of each facet, shape
        ``(N_facets, 2)``.
    cells_to_facets : np.ndarray
        Global facet index of each local facet, shape ``(K, 3)``.
    facets_to_cells : np.ndarray
        Incident cells in increasing order, padded with ``0``, shape
        ``(N_facets, 4)``.
    facet_cell_mask : np.ndarray
        ``True`` for real incidences, shape ``(N_facets, 4)``.
    trace_facets : np.ndarray
        Facets with cells from two fractures, in increasing order, shape
        ``(N_trace_facets,)``.
    trace_fracture_pairs : np.ndarray
        The two fractures of each trace facet, in increasing order, shape
        ``(N_trace_facets, 2)``.

    Raises
    ------
    ValueError
        If a facet is shared by more than two fractures, or a fracture has
        more than two cells on one facet.
    """
    number_of_cells = cells_to_vertices.shape[0]
    number_of_fractures = int(cells_to_fractures.max()) + 1

    local_facet_vertices = cells_to_vertices[..., _LOCAL_FACETS_TO_LOCAL_VERTICES]
    flat_facet_vertices = local_facet_vertices.reshape(3 * number_of_cells, 2)
    facets_to_vertices, inverse = np.unique(
        np.sort(flat_facet_vertices, axis=-1), axis=0, return_inverse=True
    )
    inverse = np.ravel(inverse)
    number_of_facets = facets_to_vertices.shape[0]
    cells_to_facets = inverse.reshape(number_of_cells, 3)

    incident_cells = np.repeat(np.arange(number_of_cells), 3)
    facet_fracture_pairs, cells_per_pair = np.unique(
        inverse * number_of_fractures + cells_to_fractures[incident_cells],
        return_counts=True,
    )
    pair_facets = facet_fracture_pairs // number_of_fractures

    overfull = pair_facets[cells_per_pair > _MAX_CELLS_PER_FRACTURE_AND_FACET]
    if overfull.shape[0] > 0:
        raise ValueError(
            "Non-manifold fracture: facet(s) with vertices "
            f"{facets_to_vertices[overfull].tolist()} have more than "
            f"{_MAX_CELLS_PER_FRACTURE_AND_FACET} cells of one fracture."
        )
    fractures_per_facet = np.bincount(pair_facets, minlength=number_of_facets)
    if np.any(fractures_per_facet > _MAX_FRACTURES_PER_FACET):
        crowded = np.flatnonzero(fractures_per_facet > _MAX_FRACTURES_PER_FACET)
        raise ValueError(
            "Facet(s) with vertices "
            f"{facets_to_vertices[crowded].tolist()} are shared by more than "
            f"{_MAX_FRACTURES_PER_FACET} fractures; every trace must belong "
            "to exactly two."
        )
    trace_facets = np.flatnonzero(fractures_per_facet == _MAX_FRACTURES_PER_FACET)
    on_trace = fractures_per_facet[pair_facets] == _MAX_FRACTURES_PER_FACET
    trace_fracture_pairs = np.reshape(
        facet_fracture_pairs[on_trace] % number_of_fractures,
        (trace_facets.shape[0], _MAX_FRACTURES_PER_FACET),
    )

    order = np.argsort(inverse, kind="stable")
    sorted_facets = inverse[order]
    cells_per_facet = np.bincount(inverse, minlength=number_of_facets)
    first_slot = np.cumsum(cells_per_facet) - cells_per_facet
    slots = np.arange(sorted_facets.shape[0]) - first_slot[sorted_facets]

    facets_to_cells = np.zeros((number_of_facets, _MAX_CELLS_PER_FACET), dtype=np.int64)
    facet_cell_mask = np.zeros((number_of_facets, _MAX_CELLS_PER_FACET), dtype=bool)
    facets_to_cells[sorted_facets, slots] = incident_cells[order]
    facet_cell_mask[sorted_facets, slots] = True

    return (
        facets_to_vertices,
        cells_to_facets,
        facets_to_cells,
        facet_cell_mask,
        trace_facets,
        trace_fracture_pairs,
    )


def _group_trace_facets(
    vertex_coordinates: np.ndarray,
    facets_to_vertices: np.ndarray,
    trace_facets: np.ndarray,
    trace_fracture_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the trace facets into traces and order them along each trace.

    Parameters
    ----------
    vertex_coordinates : np.ndarray
        Vertex coordinates, shape ``(N_vertices, 3)``.
    facets_to_vertices : np.ndarray
        Canonical vertex pair of each facet, shape ``(N_facets, 2)``.
    trace_facets : np.ndarray
        Facets with cells from two fractures, shape ``(N_trace_facets,)``.
    trace_fracture_pairs : np.ndarray
        The two fractures of each trace facet, shape ``(N_trace_facets, 2)``.

    Returns
    -------
    trace_facets : np.ndarray
        The trace facets grouped by trace and ordered along each trace,
        shape ``(N_trace_facets,)``.
    trace_facets_to_traces : np.ndarray
        Trace of each reordered trace facet, shape ``(N_trace_facets,)``.
    traces_to_fractures : np.ndarray
        The two fractures of each trace, shape ``(N_traces, 2)``.
    trace_origins : np.ndarray
        Start point of each trace, shape ``(N_traces, 3)``.
    trace_directions : np.ndarray
        Unit direction of each trace, shape ``(N_traces, 3)``.

    Notes
    -----
    The facets shared by the same two fractures lie on one line, whose
    direction is the principal direction of their vertices. They are sorted
    by the projection of their midpoints onto it and split into traces
    wherever two consecutive facets share no vertex. Traces are numbered by
    fracture pair, then along the line.
    """
    ordered_facets: list[np.ndarray] = []
    facet_traces: list[np.ndarray] = []
    traces_to_fractures: list[np.ndarray] = []
    trace_origins: list[np.ndarray] = []
    trace_directions: list[np.ndarray] = []

    unique_pairs, pair_of_facet = np.unique(
        trace_fracture_pairs, axis=0, return_inverse=True
    )
    pair_of_facet = np.ravel(pair_of_facet)
    for pair_index, pair in enumerate(unique_pairs):
        facets = trace_facets[pair_of_facet == pair_index]
        endpoints = vertex_coordinates[facets_to_vertices[facets]]
        points = np.reshape(endpoints, (-1, 3))
        _, _, right_singular_vectors = np.linalg.svd(
            points - np.mean(points, axis=0), full_matrices=False
        )
        direction = _orient_by_largest_component(right_singular_vectors[0])

        projections = endpoints @ direction
        order = np.argsort(np.mean(projections, axis=-1), kind="stable")
        facets, endpoints, projections = (
            facets[order],
            endpoints[order],
            projections[order],
        )
        facet_vertices = facets_to_vertices[facets]
        connected = np.any(
            np.expand_dims(facet_vertices[1:], axis=-1)
            == np.expand_dims(facet_vertices[:-1], axis=-2),
            axis=(-2, -1),
        )

        for segment in np.split(
            np.arange(facets.shape[0]), np.flatnonzero(~connected) + 1
        ):
            start = np.argmin(projections[segment])
            ordered_facets.append(facets[segment])
            facet_traces.append(np.full(segment.shape[0], len(trace_origins)))
            traces_to_fractures.append(pair)
            trace_origins.append(np.reshape(endpoints[segment], (-1, 3))[start])
            trace_directions.append(direction)

    if not trace_origins:
        return (
            trace_facets,
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
        )
    return (
        np.concatenate(ordered_facets),
        np.concatenate(facet_traces),
        np.stack(traces_to_fractures),
        np.stack(trace_origins),
        np.stack(trace_directions),
    )


def create_fracture_frames(mesh: DfnMesh) -> jax.Array:
    """
    Orthonormal tangent frame of every fracture plane.

    Parameters
    ----------
    mesh : DfnMesh
        The fracture network mesh.

    Returns
    -------
    frames : jax.Array
        Frame ``Q_i`` of each fracture, whose two columns are orthonormal
        vectors tangent to the plane, shape ``(N_fractures, 3, 2)``.

    Notes
    -----
    The first tangent is the coordinate axis least aligned with the normal,
    projected onto the plane and normalized. The second is
    ``normal x first``, so ``(first, second, normal)`` is right-handed. The
    frames are needed only for anisotropic transmissivities and for in-plane
    post-processing.
    """
    normals = jnp.squeeze(mesh.fracture_normals, axis=-2)
    axes = jnp.eye(3)[jnp.argmin(jnp.abs(normals), axis=-1)]
    projected = axes - jnp.sum(axes * normals, axis=-1, keepdims=True) * normals
    first = projected / jnp.linalg.norm(projected, axis=-1, keepdims=True)
    second = jnp.cross(normals, first)
    return jnp.stack((first, second), axis=-1)


def read_dfn_mesh_avs(
    inp_filepath: str | Path,
    *,
    boundary_data: BoundaryData | None = None,
    boundary_facet_tags: Mapping[tuple[int, int], int] | None = None,
    boundary_tag_names: Mapping[str, int] | None = None,
    coplanarity_tolerance: float = 1e-8,
) -> DfnMesh:
    """
    Read a fracture network mesh from an AVS UCD file.

    Parameters
    ----------
    inp_filepath : str | Path
        Path to the file, such as the ``full_mesh.inp`` written by dfnWorks.
    boundary_data : BoundaryData | None
        Boundary tagging by predicates on boundary facet midpoints.
    boundary_facet_tags : Mapping[tuple[int, int], int] | None
        Boundary tagging by canonical ``(low, high)`` vertex pair, with
        0-based vertex indices in file order.
    boundary_tag_names : Mapping[str, int] | None
        Tag name to integer value, used with ``boundary_facet_tags``.
    coplanarity_tolerance : float
        Passed to ``create_dfn_mesh_from_arrays``.

    Returns
    -------
    mesh : DfnMesh
        The mesh, with each cell's material as its fracture tag.

    Raises
    ------
    ValueError
        If the file is truncated, a cell is not a triangle, or a cell
        refers to an unknown node id.

    Notes
    -----
    The header line holds the node, cell, node-data, cell-data, and
    model-data counts. It is followed by one ``id x y z`` line per node and
    one ``id material tri n1 n2 n3`` line per cell, where ``n1 n2 n3`` are
    node ids. The data blocks after the cells are not read. dfnWorks writes
    the fracture id as the cell material; its per-node materials are not
    used, because a node on a trace carries the id of only one of its
    fractures.
    """
    lines = Path(inp_filepath).read_text().splitlines()
    header = lines[0].split() if lines else []
    if len(header) < 2:
        raise ValueError(f"{inp_filepath}: missing AVS UCD header line.")
    number_of_nodes, number_of_cells = int(header[0]), int(header[1])
    if len(lines) < 1 + number_of_nodes + number_of_cells:
        raise ValueError(
            f"{inp_filepath}: expected {number_of_nodes} node lines and "
            f"{number_of_cells} cell lines after the header."
        )

    node_rows = [line.split() for line in lines[1 : 1 + number_of_nodes]]
    cell_rows = [
        line.split()
        for line in lines[1 + number_of_nodes : 1 + number_of_nodes + number_of_cells]
    ]
    for row_index, row in enumerate(cell_rows):
        if len(row) != 6 or row[2] != "tri":
            raise ValueError(
                f"{inp_filepath}: cell line {row_index + 1} is not a triangle "
                f"'id material tri n1 n2 n3': {' '.join(row)!r}."
            )

    node_ids = np.array([int(row[0]) for row in node_rows], dtype=np.int64)
    vertex_coordinates = np.array([row[1:4] for row in node_rows], dtype=np.float64)
    cell_tags = np.array([int(row[1]) for row in cell_rows], dtype=np.int64)
    cell_node_ids = np.array([row[3:6] for row in cell_rows], dtype=np.int64)

    order = np.argsort(node_ids)
    sorted_node_ids = node_ids[order]
    positions = np.clip(
        np.searchsorted(sorted_node_ids, cell_node_ids), 0, number_of_nodes - 1
    )
    unknown = sorted_node_ids[positions] != cell_node_ids
    if np.any(unknown):
        raise ValueError(
            f"{inp_filepath}: cells refer to unknown node ids "
            f"{np.unique(cell_node_ids[unknown]).tolist()}."
        )

    return create_dfn_mesh_from_arrays(
        vertex_coordinates,
        order[positions],
        cell_tags,
        boundary_data=boundary_data,
        boundary_facet_tags=boundary_facet_tags,
        boundary_tag_names=boundary_tag_names,
        coplanarity_tolerance=coplanarity_tolerance,
    )


def create_dfn_two_fractures_mesh(cells_per_unit_length: int) -> DfnMesh:
    """
    Structured mesh of two orthogonal fractures crossing along a trace.

    Parameters
    ----------
    cells_per_unit_length : int
        Number of grid intervals ``n`` per unit length along each fracture
        edge.

    Returns
    -------
    mesh : DfnMesh
        Mesh of ``F1 = (-1, 1) x (0, 1) x {0}`` (tag ``1``) and
        ``F2 = {0} x (0, 1) x (-1, 1)`` (tag ``2``), with every boundary
        facet tagged ``"boundary"``.

    Notes
    -----
    Each fracture is a grid of ``2n x n`` squares, each split into two
    triangles along a diagonal. The fractures cross along the trace
    ``x = z = 0``, whose ``n + 1`` vertices they share, so each of its ``n``
    facets has four incident cells. Vertices are generated on an integer
    lattice and scaled by ``1 / n``, so the shared vertices coincide
    exactly.
    """
    n = cells_per_unit_length
    if n < 1:
        raise ValueError(f"cells_per_unit_length must be >= 1, got {n}.")

    long_index, short_index = np.meshgrid(
        np.arange(2 * n + 1), np.arange(n + 1), indexing="ij"
    )
    long_coordinate = np.ravel(long_index) - n
    short_coordinate = np.ravel(short_index)
    zeros = np.zeros_like(long_coordinate)
    lattice_points = np.concatenate(
        (
            np.stack((long_coordinate, short_coordinate, zeros), axis=-1),
            np.stack((zeros, short_coordinate, long_coordinate), axis=-1),
        )
    )
    unique_lattice_points, lattice_to_vertex = np.unique(
        lattice_points, axis=0, return_inverse=True
    )
    lattice_to_vertex = np.ravel(lattice_to_vertex)

    cell_long, cell_short = np.meshgrid(np.arange(2 * n), np.arange(n), indexing="ij")
    lower_left = np.ravel(cell_long * (n + 1) + cell_short)
    lower_right = lower_left + n + 1
    grid_cells = np.concatenate(
        (
            np.stack((lower_left, lower_right, lower_right + 1), axis=-1),
            np.stack((lower_left, lower_right + 1, lower_left + 1), axis=-1),
        )
    )
    points_per_fracture = (2 * n + 1) * (n + 1)
    cells_to_vertices = lattice_to_vertex[
        np.concatenate((grid_cells, grid_cells + points_per_fracture))
    ]
    cell_tags = np.repeat(np.array([1, 2]), grid_cells.shape[0])

    return create_dfn_mesh_from_arrays(
        unique_lattice_points / n, cells_to_vertices, cell_tags
    )
