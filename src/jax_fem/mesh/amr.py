"""
Adaptive mesh refinement: Dorfler marking and longest-edge bisection.

Marking (``mark_cells_by_dorfler_bulk_criterion``) is pure JAX. Refinement
(``refine_mesh_longest_edge_bisection``) is a Rivara longest-edge bisection
written as eager host code, because its recursive propagation is sequential
and data dependent.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_fem.mesh.triangle import (
    TriangleMesh,
    cell_diameters,
    create_triangle_mesh_from_arrays,
)


def mark_cells_by_dorfler_bulk_criterion(
    cell_indicators: jax.Array, theta: float = 0.5
) -> jax.Array:
    r"""
    Dorfler (bulk-chasing) marking.

    Marks the smallest set of cells :math:`\mathcal{M}` whose indicators
    carry at least a ``theta`` fraction of the total estimated error:

    .. math::

        \sum_{K \in \mathcal{M}} \eta_K^2 \ge \theta \sum_K \eta_K^2

    Parameters
    ----------
    cell_indicators : jax.Array
        Per-cell indicators ``eta_K``, shape ``(K, 1, 1, 1)`` (as returned by
        ``compute_residual_error_estimator``) or ``(K,)``.
    theta : float
        Bulk fraction, in ``(0, 1]``. Larger marks more cells; ``theta = 1``
        marks every cell (equivalent to uniform refinement).

    Returns
    -------
    marked_cells : jax.Array
        Boolean mask, shape ``(K,)``, ``True`` for marked cells.

    Notes
    -----
    The mask is flat because it feeds the host-side refinement in
    ``refine_mesh_longest_edge_bisection``.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError(f"theta must be in (0, 1], got {theta}.")

    squared_indicators = jnp.ravel(cell_indicators) ** 2
    order = jnp.flip(jnp.argsort(squared_indicators))
    cumulative = jnp.cumsum(squared_indicators[order])
    target = theta * jnp.sum(squared_indicators)

    # searchsorted on the non-decreasing cumulative sum finds the first prefix
    # that reaches the target.
    number_marked = jnp.searchsorted(cumulative, target) + 1
    number_marked = jnp.minimum(number_marked, squared_indicators.shape[-1])

    rank = jnp.argsort(order)
    return rank < number_marked


def compute_target_cell_sizes(
    mesh: TriangleMesh,
    marked_cells: jax.Array,
    refinement_factor: float = 0.5,
) -> jax.Array:
    """
    Target cell sizes for the next refinement step.

    Parameters
    ----------
    mesh : TriangleMesh
        The current mesh.
    marked_cells : jax.Array
        Boolean mask, shape ``(K,)`` (e.g. from
        ``mark_cells_by_dorfler_bulk_criterion``).
    refinement_factor : float
        Shrink factor applied to marked cells' current diameter, in
        ``(0, 1)``.

    Returns
    -------
    target_sizes : jax.Array
        Target sizes, shape ``(K,)``.

    Notes
    -----
    Marked cells get ``h_K * refinement_factor``; unmarked cells keep their
    diameter ``h_K``.
    """
    if not (0.0 < refinement_factor < 1.0):
        raise ValueError(
            f"refinement_factor must be in (0, 1), got {refinement_factor}."
        )
    current_sizes = cell_diameters(mesh)
    return jnp.where(marked_cells, current_sizes * refinement_factor, current_sizes)


def _canonical_edge(vertex_a: int, vertex_b: int) -> tuple[int, int]:
    """
    Order a vertex pair as ``(low, high)``.

    Parameters
    ----------
    vertex_a : int
        First vertex index.
    vertex_b : int
        Second vertex index.

    Returns
    -------
    edge : tuple[int, int]
        The two indices in increasing order.
    """
    return (vertex_a, vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)


def _cell_longest_edge(
    cell: tuple[int, int, int], vertices: list[tuple[float, float]]
) -> tuple[int, int, int]:
    """
    Longest edge of a cell and its opposite vertex.

    Parameters
    ----------
    cell : tuple[int, int, int]
        Vertex indices of the cell.
    vertices : list[tuple[float, float]]
        Vertex coordinates.

    Returns
    -------
    rotation : tuple[int, int, int]
        ``(opposite, edge_start, edge_end)``, a cyclic rotation of ``cell``
        whose last two entries span the longest edge, in the cell's own
        orientation.
    """
    v0, v1, v2 = cell
    rotations = ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1))
    lengths = [math.dist(vertices[a], vertices[b]) for _, a, b in rotations]
    return rotations[max(range(3), key=lengths.__getitem__)]


def refine_mesh_longest_edge_bisection(
    mesh: TriangleMesh, marked_cells: jax.Array
) -> TriangleMesh:
    r"""
    Rivara longest-edge bisection of the marked cells.

    Parameters
    ----------
    mesh : TriangleMesh
        The mesh to refine; it is not modified.
    marked_cells : jax.Array
        Boolean mask of cells to refine, shape ``(K,)``.

    Returns
    -------
    refined_mesh : TriangleMesh
        A new refined mesh, or ``mesh`` itself if no cell is marked.

    Notes
    -----
    Each marked cell is split along its longest edge. When the shared edge
    is not also the neighbour's longest edge, the neighbour is refined first,
    which keeps the mesh conforming (no hanging nodes) and shape regular
    (Rivara, 1984). The propagation uses an explicit stack, so long chains
    cannot hit Python's recursion limit. Boundary and cell tags are passed
    on to the children. The routine is eager host code, not traceable by
    ``jax.jit``.
    """
    return _refine_mesh_longest_edge_bisection(
        mesh, marked_cells, max_bisections=None
    )


def _refine_mesh_longest_edge_bisection(
    mesh: TriangleMesh, marked_cells: jax.Array, *, max_bisections: int | None
) -> TriangleMesh:
    """
    Longest-edge bisection with an optional cap on the number of bisections.

    Parameters
    ----------
    mesh : TriangleMesh
        The mesh to refine.
    marked_cells : jax.Array
        Boolean mask of cells to refine, shape ``(K,)``.
    max_bisections : int | None
        Bisection cap; ``None`` uses ``4 K``.

    Returns
    -------
    refined_mesh : TriangleMesh
        The refined mesh, or ``mesh`` if no cell is marked.

    Notes
    -----
    The cap is a defensive termination guard; exposing it lets the guard be
    tested without a mesh large enough to trigger it.
    """
    marked_indices = [int(i) for i in np.flatnonzero(np.asarray(marked_cells))]
    if not marked_indices:
        return mesh

    vertices: list[tuple[float, float]] = [
        (float(x), float(y))
        for x, y in np.asarray(jnp.squeeze(mesh.vertex_coordinates, axis=-2))
    ]
    cells: list[tuple[int, int, int] | None] = [
        tuple(int(v) for v in row) for row in np.asarray(mesh.cells_to_vertices)
    ]
    cell_tags: list[int] | None = (
        [int(t) for t in np.asarray(mesh.cell_tags)]
        if mesh.cell_tags is not None
        else None
    )

    boundary_tag_names = dict(mesh.boundary_tag_names)
    edge_to_boundary_tag: dict[tuple[int, int], int] = {}
    facets_to_vertices_np = np.asarray(mesh.facets_to_vertices)
    boundary_facet_tags_np = np.asarray(mesh.boundary_facet_tags)
    for row, facet_index in enumerate(np.asarray(mesh.boundary_facets)):
        vertex_a, vertex_b = (int(v) for v in facets_to_vertices_np[facet_index])
        edge_to_boundary_tag[_canonical_edge(vertex_a, vertex_b)] = int(
            boundary_facet_tags_np[row]
        )

    edge_to_midpoint_id: dict[tuple[int, int], int] = {}
    edge_to_cells: dict[tuple[int, int], list[int]] = {}

    def _cell_edges(cell: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
        """
        Canonical edges of a cell, ordered by opposite local vertex.

        Parameters
        ----------
        cell : tuple[int, int, int]
            Vertex indices of the cell.

        Returns
        -------
        edges : tuple[tuple[int, int], ...]
            The three canonical edges.
        """
        v0, v1, v2 = cell
        return (
            _canonical_edge(v1, v2),
            _canonical_edge(v2, v0),
            _canonical_edge(v0, v1),
        )

    for cell_index, cell in enumerate(cells):
        for edge in _cell_edges(cell):
            edge_to_cells.setdefault(edge, []).append(cell_index)

    def _find_neighbor(cell_index: int, edge: tuple[int, int]) -> int | None:
        """
        The other cell sharing an edge.

        Parameters
        ----------
        cell_index : int
            The cell whose neighbour is wanted.
        edge : tuple[int, int]
            Canonical edge shared with the neighbour.

        Returns
        -------
        neighbor : int | None
            The neighbouring cell, or ``None`` on the boundary.
        """
        others = [c for c in edge_to_cells.get(edge, ()) if c != cell_index]
        return others[0] if others else None

    bisected_cells: set[int] = set()
    bound = 4 * len(cells) if max_bisections is None else max_bisections
    bisection_count = 0

    def _bisect(cell_index: int) -> None:
        """
        Split a cell in two along its longest edge.

        Parameters
        ----------
        cell_index : int
            The cell to split; it is retired and replaced by two children.

        Raises
        ------
        RuntimeError
            If the number of bisections exceeds the cap.
        """
        nonlocal bisection_count
        cell = cells[cell_index]
        assert cell is not None
        opposite, edge_start, edge_end = _cell_longest_edge(cell, vertices)
        edge_key = _canonical_edge(edge_start, edge_end)

        midpoint_index = edge_to_midpoint_id.get(edge_key)
        if midpoint_index is None:
            xa, ya = vertices[edge_start]
            xb, yb = vertices[edge_end]
            midpoint_index = len(vertices)
            vertices.append(((xa + xb) / 2.0, (ya + yb) / 2.0))
            edge_to_midpoint_id[edge_key] = midpoint_index

        child_1 = (opposite, edge_start, midpoint_index)
        child_2 = (opposite, midpoint_index, edge_end)

        for edge in _cell_edges(cell):
            edge_to_cells[edge].remove(cell_index)
        cells[cell_index] = None

        tag = cell_tags[cell_index] if cell_tags is not None else None
        for child in (child_1, child_2):
            child_index = len(cells)
            cells.append(child)
            if cell_tags is not None:
                cell_tags.append(tag)
            for edge in _cell_edges(child):
                edge_to_cells.setdefault(edge, []).append(child_index)

        bisected_cells.add(cell_index)
        bisection_count += 1
        if bisection_count > bound:
            raise RuntimeError(
                "LEB refinement loop exceeded its maximum bisection bound "
                f"({bound}); this indicates a bug in the propagation logic "
                "rather than a legitimate refinement, since Rivara "
                "bisection is proven to terminate."
            )

        boundary_tag = edge_to_boundary_tag.pop(edge_key, None)
        if boundary_tag is not None:
            edge_to_boundary_tag[_canonical_edge(edge_start, midpoint_index)] = (
                boundary_tag
            )
            edge_to_boundary_tag[_canonical_edge(midpoint_index, edge_end)] = (
                boundary_tag
            )

    for seed_index in marked_indices:
        stack = [seed_index]
        while stack:
            current_index = stack[-1]
            if current_index in bisected_cells:
                stack.pop()
                continue

            current_cell = cells[current_index]
            assert current_cell is not None
            _, edge_start, edge_end = _cell_longest_edge(current_cell, vertices)
            edge_key = _canonical_edge(edge_start, edge_end)
            neighbor_index = _find_neighbor(current_index, edge_key)

            neighbor_needs_matching_bisect = False
            if neighbor_index is not None and neighbor_index not in bisected_cells:
                neighbor_cell = cells[neighbor_index]
                assert neighbor_cell is not None
                _, neighbor_start, neighbor_end = _cell_longest_edge(
                    neighbor_cell, vertices
                )
                if _canonical_edge(neighbor_start, neighbor_end) != edge_key:
                    # The shared edge is not the neighbour's longest edge:
                    # refine the neighbour first.
                    stack.append(neighbor_index)
                    continue
                neighbor_needs_matching_bisect = True

            stack.pop()
            _bisect(current_index)
            if neighbor_needs_matching_bisect:
                _bisect(neighbor_index)

    final_cells = [cell for cell in cells if cell is not None]
    final_cell_tags = (
        [tag for cell, tag in zip(cells, cell_tags, strict=True) if cell is not None]
        if cell_tags is not None
        else None
    )

    return create_triangle_mesh_from_arrays(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(final_cells, dtype=np.int64),
        boundary_facet_tags=edge_to_boundary_tag,
        boundary_tag_names=boundary_tag_names,
        cell_tags=(
            np.asarray(final_cell_tags, dtype=np.int64)
            if final_cell_tags is not None
            else None
        ),
    )


def adapt_mesh_from_error_estimator(
    mesh: TriangleMesh, cell_indicators: jax.Array, theta: float = 0.5
) -> TriangleMesh:
    r"""
    One adaptive step: Dorfler marking followed by longest-edge bisection.

    Parameters
    ----------
    mesh : TriangleMesh
        The mesh the indicators were computed on; it is not modified.
    cell_indicators : jax.Array
        Per-cell indicators ``eta_K``, shape ``(K, 1, 1, 1)`` or ``(K,)``.
    theta : float
        Dorfler bulk fraction, in ``(0, 1]``.

    Returns
    -------
    refined_mesh : TriangleMesh
        The refined mesh.
    """
    marked_cells = mark_cells_by_dorfler_bulk_criterion(cell_indicators, theta)
    return refine_mesh_longest_edge_bisection(mesh, marked_cells)
