"""
Adaptive mesh refinement (AMR) driven by the residual error estimator.

Given per-cell indicators ``eta_K`` from
``jax_fem.diagnostics.compute_residual_error_estimator``, mark cells for
refinement (Dorfler bulk criterion) and refine the marked region with an
in-house Rivara Longest Edge Bisection (LEB) engine -- a pure
``TriangleMesh`` -> ``TriangleMesh`` transformation operating directly on
host vertex/connectivity arrays, with no external meshing engine involved.
See ``context/in_house_amr_implementation_plan.md`` for the full
algorithmic derivation.

Split into a pure-JAX marking layer (``mark_cells_by_dorfler_bulk_criterion``,
unit-testable on its own) and an eager-NumPy refinement engine
(``refine_mesh_longest_edge_bisection``): mesh topology mutation here is
inherently sequential and data-dependent (recursive longest-edge
propagation), so it runs as ordinary host code rather than under
``jax.jit``, mirroring the eager/host style already used for Gmsh import
elsewhere in ``jax_fem.mesh``.
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
        Per-cell indicators ``eta_K``, shape ``(K,)`` (e.g. from
        ``compute_residual_error_estimator``).
    theta : float
        Bulk fraction, in ``(0, 1]``. Larger marks more cells; ``theta = 1``
        marks every cell (equivalent to uniform refinement).

    Returns
    -------
    jax.Array
        Boolean mask, shape ``(K,)``, ``True`` for marked cells.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError(f"theta must be in (0, 1], got {theta}.")

    squared_indicators = cell_indicators**2
    order = jnp.argsort(squared_indicators)[::-1]  # descending
    cumulative = jnp.cumsum(squared_indicators[order])
    target = theta * jnp.sum(squared_indicators)

    # Smallest prefix length M such that cumulative[:M] >= target (searchsorted
    # on a non-decreasing array finds the first index whose value is >= target).
    number_marked = jnp.searchsorted(cumulative, target) + 1
    number_marked = jnp.minimum(number_marked, cell_indicators.shape[0])

    rank = jnp.argsort(order)  # rank[i] = position of cell i in the sorted order
    return rank < number_marked


def compute_target_cell_sizes(
    mesh: TriangleMesh,
    marked_cells: jax.Array,
    refinement_factor: float = 0.5,
) -> jax.Array:
    """
    Target per-cell element size for the next AMR iteration.

    Marked cells shrink to ``h_K * refinement_factor``; unmarked cells keep
    their current diameter ``h_K`` (``jax_fem.mesh.triangle.cell_diameters``).

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
    jax.Array
        Target sizes, shape ``(K,)``.
    """
    if not (0.0 < refinement_factor < 1.0):
        raise ValueError(
            f"refinement_factor must be in (0, 1), got {refinement_factor}."
        )
    current_sizes = cell_diameters(mesh)
    return jnp.where(marked_cells, current_sizes * refinement_factor, current_sizes)


def _canonical_edge(vertex_a: int, vertex_b: int) -> tuple[int, int]:
    """Sort a vertex-index pair into ``(low, high)`` canonical order."""
    return (vertex_a, vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)


def _cell_longest_edge(
    cell: tuple[int, int, int], vertices: list[tuple[float, float]]
) -> tuple[int, int, int]:
    """
    A cell's opposite-vertex/longest-edge decomposition.

    Returns ``(opposite, edge_start, edge_end)``: a cyclic rotation of
    ``cell`` (so still in ``cell``'s own counter-clockwise winding order)
    with ``opposite`` the vertex not on the longest edge and
    ``(edge_start, edge_end)`` its two endpoints, traversed in ``cell``'s
    own orientation.
    """
    v0, v1, v2 = cell
    rotations = ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1))
    lengths = [math.dist(vertices[a], vertices[b]) for _, a, b in rotations]
    return rotations[max(range(3), key=lengths.__getitem__)]


def refine_mesh_longest_edge_bisection(
    mesh: TriangleMesh, marked_cells: jax.Array
) -> TriangleMesh:
    r"""
    Rivara Longest Edge Bisection (LEB) refinement of the marked cells.

    A pure ``TriangleMesh`` -> ``TriangleMesh`` transformation: every marked
    cell is split in two along its longest edge, with recursive
    ("matching") propagation into neighbors whenever the shared edge is not
    also *their* longest edge -- the classical construction that guarantees
    strict :math:`H^1` conformity (zero hanging nodes) and preserves shape
    regularity (Rivara, 1984). ``mesh`` itself is never mutated.

    Runs as eager host NumPy code: the propagation loop is sequential and
    data-dependent, not a shape suited to ``jax.jit``/``vmap``. An explicit
    LIFO stack (rather than Python recursion) drives the propagation, so
    long chains cannot raise ``RecursionError``.

    Boundary facet tags and physical cell tags are propagated to child
    cells/sub-facets; vertices/cells untouched by refinement are carried
    through unchanged.

    Parameters
    ----------
    mesh : TriangleMesh
        The mesh to refine.
    marked_cells : jax.Array
        Boolean mask, shape ``(K,)`` (e.g. from
        ``mark_cells_by_dorfler_bulk_criterion``), ``True`` for cells to
        refine.

    Returns
    -------
    TriangleMesh
        A new, independent, refined ``TriangleMesh``. If no cell is marked,
        ``mesh`` is returned unchanged.
    """
    return _refine_mesh_longest_edge_bisection(
        mesh, marked_cells, max_bisections=None
    )


def _refine_mesh_longest_edge_bisection(
    mesh: TriangleMesh, marked_cells: jax.Array, *, max_bisections: int | None
) -> TriangleMesh:
    """
    ``refine_mesh_longest_edge_bisection``, with an overridable bisection
    cap -- factored out so the defensive termination guard is directly
    unit-testable without needing a mesh large enough to trigger it
    naturally (which the algorithm's own termination proof makes
    impractical to construct).
    """
    marked_indices = [int(i) for i in np.flatnonzero(np.asarray(marked_cells))]
    if not marked_indices:
        return mesh

    vertices: list[tuple[float, float]] = [
        (float(x), float(y))
        for x, y in np.asarray(mesh.vertex_coordinates[:, 0, :])
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
        others = [c for c in edge_to_cells.get(edge, ()) if c != cell_index]
        return others[0] if others else None

    bisected_cells: set[int] = set()
    bound = 4 * len(cells) if max_bisections is None else max_bisections
    bisection_count = 0

    def _bisect(cell_index: int) -> None:
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
        cells[cell_index] = None  # retire; never referenced again

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
                    # e is not neighbor's own longest edge: resolve it first
                    # (recursive propagation via the explicit stack).
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
    One adaptive refinement iteration: Dorfler-mark, then LEB-bisect.

    Fully domain-agnostic: ``refine_mesh_longest_edge_bisection`` is a pure
    ``TriangleMesh`` -> ``TriangleMesh`` transformation with no reference to
    how ``mesh`` was originally built, so this works identically for the
    unit square, the L-shaped domain, or any other mesh.

    Parameters
    ----------
    mesh : TriangleMesh
        The mesh ``cell_indicators`` was computed on.
    cell_indicators : jax.Array
        Per-cell error indicators ``eta_K``, shape ``(K,)`` (e.g. from
        ``compute_residual_error_estimator``).
    theta : float
        Dorfler bulk fraction, in ``(0, 1]``.

    Returns
    -------
    TriangleMesh
        A new, independent, refined ``TriangleMesh``; ``mesh`` is left
        untouched.
    """
    marked_cells = mark_cells_by_dorfler_bulk_criterion(cell_indicators, theta)
    return refine_mesh_longest_edge_bisection(mesh, marked_cells)
