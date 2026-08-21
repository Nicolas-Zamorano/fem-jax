"""
Adaptive mesh refinement (AMR) driven by the residual error estimator.

See Section 4.2 of the implementation plan (the AMR stretch item): given
per-cell indicators ``eta_K`` from
``jax_fem.diagnostics.compute_residual_error_estimator``, mark cells for
refinement, compute a new spatially-varying target mesh size, and rebuild
the mesh via a Gmsh background size field. Split into a pure-JAX layer
(marking, sizing -- unit-testable without Gmsh at all) and a Gmsh-only
bridge layer, mirroring the separation already used between
``jax_fem.diagnostics.estimator`` (pure) and ``jax_fem.mesh.gmsh`` (impure)
elsewhere in this library.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_fem.mesh.gmsh import l_shaped_gmsh_geometry, run_in_owned_gmsh_session
from jax_fem.mesh.triangle import TriangleMesh, cell_diameters


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


def _build_postview_size_field(
    gmsh_module: Any, mesh: TriangleMesh, target_cell_sizes: jax.Array
) -> None:
    """
    Install a Gmsh ``PostView``-backed background mesh size field.

    Gmsh's list-based scalar-triangle view data (``"ST"``) is *not*
    centroid+value per triangle: it is 12 floats per triangle -- the 3
    corner coordinates (9 floats) followed by a scalar value at each of the
    3 corners (3 floats, here the same ``target_cell_sizes[k]`` repeated).
    Coordinates come straight from ``mesh``, so this needs no reference back
    to the CAD entities that built ``mesh`` -- the field is purely
    coordinate-based.

    Parameters
    ----------
    gmsh_module:
        The (already-initialized) ``gmsh`` module, mid-geometry-construction
        (called as a ``size_field_builder`` from ``l_shaped_gmsh_geometry``,
        after physical groups are tagged and before meshing).
    mesh : TriangleMesh
        The *previous* iteration's mesh that ``target_cell_sizes`` is
        defined over.
    target_cell_sizes : jax.Array
        Shape ``(K,)``, matching ``mesh.cells_to_vertices.shape[0]``.
    """
    coordinates = np.asarray(mesh.vertex_coordinates[:, 0, :])  # (N, 2)
    triangle_vertex_indices = np.asarray(mesh.cells_to_vertices)  # (K, 3)
    triangle_coordinates = coordinates[triangle_vertex_indices]  # (K, 3, 2)
    number_of_cells = triangle_coordinates.shape[0]

    zeros = np.zeros((number_of_cells, 3, 1))
    corners = np.concatenate([triangle_coordinates, zeros], axis=-1)  # (K, 3, 3)

    sizes = np.asarray(target_cell_sizes).reshape(number_of_cells, 1)
    values = np.broadcast_to(sizes, (number_of_cells, 3))  # (K, 3)

    st_data = np.concatenate(
        [corners.reshape(number_of_cells, 9), values], axis=1
    ).ravel()  # (12 * K,)

    view_tag = gmsh_module.view.add("amr_target_size")
    gmsh_module.view.addListData(view_tag, "ST", number_of_cells, st_data.tolist())

    field_tag = gmsh_module.model.mesh.field.add("PostView")
    gmsh_module.model.mesh.field.setNumber(field_tag, "ViewTag", view_tag)
    gmsh_module.model.mesh.field.setAsBackgroundMesh(field_tag)


def adapt_l_shaped_mesh_from_error_estimator(
    mesh: TriangleMesh,
    cell_indicators: jax.Array,
    *,
    theta: float = 0.5,
    refinement_factor: float = 0.5,
) -> TriangleMesh:
    """
    One AMR iteration for the re-entrant L-shaped domain.

    Marks cells via ``mark_cells_by_dorfler_bulk_criterion``, computes a new
    target size field via ``compute_target_cell_sizes``, and rebuilds the
    L-shaped domain (``l_shaped_gmsh_geometry``) from scratch in a freshly
    owned Gmsh session with that field as the background mesh size --
    ``mesh`` itself is never mutated, and the returned ``TriangleMesh`` is
    otherwise the same kind of pure, immutable snapshot every other factory
    in this module returns.

    Parameters
    ----------
    mesh : TriangleMesh
        The mesh ``cell_indicators`` was computed on (e.g. via
        ``create_gmsh_l_shaped_mesh`` or a previous call to this function).
    cell_indicators : jax.Array
        Per-cell error indicators ``eta_K``, shape ``(K,)`` (from
        ``compute_residual_error_estimator``).
    theta : float
        Dorfler bulk fraction, in ``(0, 1]``.
    refinement_factor : float
        Shrink factor applied to marked cells' current diameter, in
        ``(0, 1)``.

    Returns
    -------
    TriangleMesh
        A new, independently meshed ``TriangleMesh`` of the L-shaped domain,
        graded toward the marked cells.
    """
    marked_cells = mark_cells_by_dorfler_bulk_criterion(cell_indicators, theta)
    target_cell_sizes = compute_target_cell_sizes(
        mesh, marked_cells, refinement_factor
    )

    def size_field_builder(gmsh_module: Any) -> None:
        _build_postview_size_field(gmsh_module, mesh, target_cell_sizes)

    return run_in_owned_gmsh_session(
        lambda gmsh_module: l_shaped_gmsh_geometry(
            gmsh_module, size_field_builder=size_field_builder
        )
    )
