"""
Gmsh model and ``.msh`` file import paths, and in-memory Gmsh domain factories.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_fem.mesh.triangle import TriangleMesh, create_triangle_mesh_from_arrays

# Gmsh's fixed elementary type codes (stable across Gmsh versions):
# https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
_TRIANGLE_ELEMENT_TYPE = 2  # 3-node triangle
_LINE_ELEMENT_TYPE = 1  # 2-node line

# Vertices of the re-entrant-corner L-shaped domain [-1,1]^2 \ [0,1]x[-1,0],
# listed counter-clockwise starting at the re-entrant corner (the origin).
_L_SHAPE_VERTICES = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (0.0, -1.0),
)


def _import_gmsh() -> Any:
    """Import and return the ``gmsh`` module, with a friendly error if missing."""
    try:
        import gmsh
    except ImportError as error:
        raise ImportError(
            "This function requires the 'gmsh' package (a core dependency of "
            "jax-fem). Install it with: pip install gmsh."
        ) from error
    return gmsh


def create_triangle_mesh_from_gmsh_model(model: Any) -> TriangleMesh:
    """
    Build a ``TriangleMesh`` from a live, already-meshed Gmsh model.

    The model must already have a linear (order-1) 2D mesh generated, e.g.
    via ``gmsh.model.mesh.generate(2)``. Boundary facets are tagged from the
    model's 1D physical groups (physical curves), matched **exactly** against
    mesh facets via canonical integer node-index pairs (no coordinate
    tolerance involved): a physical curve's line elements are remapped
    through the same node-tag-to-vertex-index table used for triangle
    connectivity, canonicalized as ``(min(index_a, index_b),
    max(index_a, index_b))``, and looked up directly against
    ``facets_to_vertices``. If the model has no 1D physical groups, every
    boundary facet receives the single default tag (Section 7.3). If the
    model has 2D physical groups (physical surfaces), each cell's
    ``cell_tags`` entry is set to the raw Gmsh physical tag of the surface it
    belongs to, or ``-1`` if untagged.

    Parameters
    ----------
    model:
        An object with the same interface as the ``gmsh.model`` submodule
        (``mesh.getNodes``, ``mesh.getElementsByType``, ``getPhysicalGroups``,
        ``getPhysicalName``, ``getEntitiesForPhysicalGroup``). Once imported,
        the returned ``TriangleMesh`` is independent of the live model.

    Returns
    -------
    TriangleMesh
    """
    element_tags, triangle_node_tags = _get_triangle_elements(model)
    tag_to_coords = _get_node_coordinates(model)

    used_node_tags = np.unique(triangle_node_tags)
    node_tag_to_index = {int(tag): index for index, tag in enumerate(used_node_tags)}

    vertex_coordinates = np.array(
        [tag_to_coords[int(tag)] for tag in used_node_tags]
    )
    cells_to_vertices = np.vectorize(node_tag_to_index.__getitem__)(
        triangle_node_tags
    )

    mesh = create_triangle_mesh_from_arrays(vertex_coordinates, cells_to_vertices)

    exact_tags = _exact_boundary_tags_from_physical_curves(
        model, node_tag_to_index, mesh
    )
    if exact_tags is not None:
        boundary_facet_tags_np, boundary_tag_names = exact_tags
        mesh = dataclasses.replace(
            mesh,
            boundary_facet_tags=jnp.asarray(boundary_facet_tags_np, dtype=jnp.int32),
            boundary_tag_names=boundary_tag_names,
        )

    cell_tags_np = _cell_tags_from_physical_surfaces(model, element_tags)
    if cell_tags_np is not None:
        mesh = dataclasses.replace(
            mesh, cell_tags=jnp.asarray(cell_tags_np, dtype=jnp.int32)
        )

    return mesh


def read_triangle_mesh_from_msh(path: str | os.PathLike[str]) -> TriangleMesh:
    """
    Read a ``TriangleMesh`` directly from a Gmsh ``.msh`` file.

    Parameters
    ----------
    path: str | os.PathLike[str]
        Path to a ``.msh`` file.

    Returns
    -------
    TriangleMesh
    """
    gmsh = _import_gmsh()

    already_initialized = bool(gmsh.isInitialized())
    if not already_initialized:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(os.fspath(path))
        return create_triangle_mesh_from_gmsh_model(gmsh.model)
    finally:
        if not already_initialized:
            gmsh.finalize()


def _get_triangle_elements(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Get triangle elements from a Gmsh model.

    Parameters
    ----------
    model
        Gmsh model.

    Returns
    -------
    tags : tuple[np.ndarray, np.ndarray]
        Element tags and node tags.
    """
    element_tags, node_tags_flat = model.mesh.getElementsByType(
        _TRIANGLE_ELEMENT_TYPE
    )
    element_tags = np.asarray(element_tags, dtype=np.int64)
    if element_tags.shape[0] == 0:
        raise ValueError(
            "The Gmsh model has no 3-node triangle elements. Call "
            "gmsh.model.mesh.generate(2) with a linear (order-1) mesh "
            "before importing (Section 7.4)."
        )
    triangle_node_tags = np.asarray(node_tags_flat, dtype=np.int64).reshape(-1, 3)
    return element_tags, triangle_node_tags


def _get_node_coordinates(model: Any) -> dict[int, tuple[float, float]]:
    """
    Get node coordinates from a Gmsh model.

    Parameters
    ----------
    model
        Gmsh model.

    Returns
    -------
    tag_to_coords : dict[int, tuple[float, float]]
        Dictionary mapping node tags to coordinates.
    """
    node_tags, node_coords_flat, _ = model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coords = np.asarray(node_coords_flat, dtype=np.float64).reshape(-1, 3)

    if node_coords.shape[0] > 0 and not np.allclose(node_coords[:, 2], 0.0):
        raise ValueError(
            "The Gmsh model is not planar (found nonzero z coordinates). "
            "Only 2D straight-sided triangular meshes are supported "
            "(Section 2.1)."
        )

    return {
        int(tag): (float(coord[0]), float(coord[1]))
        for tag, coord in zip(node_tags, node_coords, strict=True)
    }


def _exact_boundary_tags_from_physical_curves(
    model: Any,
    node_tag_to_index: dict[int, int],
    mesh: TriangleMesh,
) -> tuple[np.ndarray, dict[str, int]] | None:
    """
    Tag ``mesh``'s boundary facets from the model's 1D physical curves.

    Matches each physical curve's line elements against boundary facets by
    exact canonical integer vertex-index pair -- both sides derive from the
    same ``node_tag_to_index`` remap, so this is an exact lookup, not a
    coordinate-tolerance comparison.

    Parameters
    ----------
    model:
        The Gmsh model.
    node_tag_to_index:
        Gmsh node tag -> this mesh's compact vertex index, as built by
        ``create_triangle_mesh_from_gmsh_model``.
    mesh:
        The already-constructed ``TriangleMesh`` (with default boundary
        tagging) whose facets are to be retagged.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]] | None
        ``(boundary_facet_tags, boundary_tag_names)`` if the model has any
        1D physical groups, else ``None``.
    """
    physical_groups = model.getPhysicalGroups(dim=1)
    if not physical_groups:
        return None

    edge_to_name: dict[tuple[int, int], str] = {}
    boundary_tag_names: dict[str, int] = {}
    for index, (dim, physical_tag) in enumerate(physical_groups):
        name = model.getPhysicalName(dim, physical_tag) or f"boundary_{physical_tag}"
        boundary_tag_names[name] = index
        curve_tags = model.getEntitiesForPhysicalGroup(dim, physical_tag)

        for curve_tag in curve_tags:
            _, line_node_tags_flat = model.mesh.getElementsByType(
                _LINE_ELEMENT_TYPE, tag=curve_tag
            )
            line_node_tags = np.asarray(line_node_tags_flat, dtype=np.int64).reshape(
                -1, 2
            )
            for node_a, node_b in line_node_tags:
                index_a = node_tag_to_index[int(node_a)]
                index_b = node_tag_to_index[int(node_b)]
                canonical = (min(index_a, index_b), max(index_a, index_b))
                edge_to_name[canonical] = name

    boundary_facets = np.asarray(mesh.boundary_facets)
    facets_to_vertices = np.asarray(mesh.facets_to_vertices)  # already (low, high)
    boundary_facet_vertex_pairs = facets_to_vertices[boundary_facets]

    tags = np.full(boundary_facets.shape[0], -1, dtype=np.int64)
    for row, (vertex_a, vertex_b) in enumerate(boundary_facet_vertex_pairs):
        name = edge_to_name.get((int(vertex_a), int(vertex_b)))
        if name is not None:
            tags[row] = boundary_tag_names[name]

    if np.any(tags == -1):
        unmatched = boundary_facets[tags == -1]
        raise ValueError(
            f"Boundary facets {unmatched.tolist()} matched no physical curve."
        )

    return tags, boundary_tag_names


def _cell_tags_from_physical_surfaces(
    model: Any, element_tags: np.ndarray
) -> np.ndarray | None:
    physical_groups = model.getPhysicalGroups(dim=2)
    if not physical_groups:
        return None

    element_tag_to_physical_tag: dict[int, int] = {}
    for dim, physical_tag in physical_groups:
        surface_tags = model.getEntitiesForPhysicalGroup(dim, physical_tag)
        for surface_tag in surface_tags:
            surface_element_tags, _ = model.mesh.getElementsByType(
                _TRIANGLE_ELEMENT_TYPE, tag=surface_tag
            )
            for element_tag in surface_element_tags:
                element_tag_to_physical_tag[int(element_tag)] = physical_tag

    return np.array(
        [element_tag_to_physical_tag.get(int(tag), -1) for tag in element_tags],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# In-memory geometry builders: construct + mesh a domain in the *current*,
# already-initialized Gmsh session, leaving the session open. Each is usable
# either directly (see ``create_gmsh_unit_square_mesh`` /
# ``create_gmsh_l_shaped_mesh``, which own the session lifecycle around one
# call) or as the ``geometry_builder`` passed to ``create_gmsh_mesh_hierarchy``,
# which owns the session across several ``gmsh.model.mesh.refine()`` calls.
# ---------------------------------------------------------------------------


def unit_square_gmsh_geometry(gmsh_module: Any, number_of_cells_per_side: int) -> None:
    """
    Build and mesh the unit square ``[0, 1]^2`` in the current Gmsh session.

    Uses the ``geo`` kernel. Tags physical curves ``"bottom"`` (``y = 0``),
    ``"right"`` (``x = 1``), ``"top"`` (``y = 1``), ``"left"`` (``x = 0``),
    and physical surface ``"domain"``.

    Parameters
    ----------
    gmsh_module:
        The (already-initialized) ``gmsh`` module.
    number_of_cells_per_side:
        Target number of cells along each side, at least 1; converted to a
        target mesh element size ``1 / number_of_cells_per_side``.
    """
    if number_of_cells_per_side < 1:
        raise ValueError(
            "number_of_cells_per_side must be at least 1, got "
            f"{number_of_cells_per_side}."
        )
    mesh_size = 1.0 / number_of_cells_per_side

    gmsh_module.model.add("unit_square")
    p1 = gmsh_module.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
    p2 = gmsh_module.model.geo.addPoint(1.0, 0.0, 0.0, mesh_size)
    p3 = gmsh_module.model.geo.addPoint(1.0, 1.0, 0.0, mesh_size)
    p4 = gmsh_module.model.geo.addPoint(0.0, 1.0, 0.0, mesh_size)
    bottom = gmsh_module.model.geo.addLine(p1, p2)
    right = gmsh_module.model.geo.addLine(p2, p3)
    top = gmsh_module.model.geo.addLine(p3, p4)
    left = gmsh_module.model.geo.addLine(p4, p1)
    loop = gmsh_module.model.geo.addCurveLoop([bottom, right, top, left])
    surface = gmsh_module.model.geo.addPlaneSurface([loop])
    gmsh_module.model.geo.synchronize()

    gmsh_module.model.addPhysicalGroup(1, [bottom], name="bottom")
    gmsh_module.model.addPhysicalGroup(1, [right], name="right")
    gmsh_module.model.addPhysicalGroup(1, [top], name="top")
    gmsh_module.model.addPhysicalGroup(1, [left], name="left")
    gmsh_module.model.addPhysicalGroup(2, [surface], name="domain")

    gmsh_module.model.mesh.generate(2)


def l_shaped_gmsh_geometry(
    gmsh_module: Any,
    target_h: float | None = None,
    *,
    size_field_builder: Callable[[Any], None] | None = None,
) -> None:
    """
    Build and mesh the re-entrant L-shaped domain in the current Gmsh session.

    Domain: ``[-1, 1]^2 \\ [0, 1] x [-1, 0]``, with a genuine 270-degree
    re-entrant corner at the origin. Uses the ``occ`` kernel. Tags a single
    physical curve ``"boundary"`` covering the whole boundary, and physical
    surface ``"domain"``.

    Parameters
    ----------
    gmsh_module:
        The (already-initialized) ``gmsh`` module.
    target_h:
        Target mesh element size, greater than 0, used as a uniform
        per-point mesh size. Required when ``size_field_builder`` is not
        given; ignored (points left at their unconstrained default, ``0.0``)
        when it is -- see ``size_field_builder`` below.
    size_field_builder:
        Optional callback ``(gmsh_module) -> None``, invoked after physical
        groups are tagged and before ``gmsh_module.model.mesh.generate(2)``,
        to install a background mesh size field (e.g.
        ``jax_fem.mesh.amr``'s adaptive size field). When given, CAD points
        are left at mesh size ``0.0`` (Gmsh's "unconstrained" sentinel) and
        ``Mesh.MeshSizeFromPoints`` / ``Mesh.MeshSizeExtendFromBoundary`` are
        both disabled, so the field is the *sole* authority over element
        sizes -- otherwise Gmsh takes the minimum of point-prescribed and
        field-prescribed sizes at every location, which can silently keep a
        coarse point size from being refined away, including at the
        re-entrant corner itself (a CAD point here).
    """
    if size_field_builder is None:
        if target_h is None or target_h <= 0.0:
            raise ValueError(
                "target_h must be positive when size_field_builder is not "
                f"given, got {target_h}."
            )
        point_mesh_size = target_h
    else:
        point_mesh_size = 0.0

    gmsh_module.model.add("l_shape")
    point_tags = [
        gmsh_module.model.occ.addPoint(x, y, 0.0, point_mesh_size)
        for x, y in _L_SHAPE_VERTICES
    ]
    number_of_vertices = len(point_tags)
    line_tags = [
        gmsh_module.model.occ.addLine(
            point_tags[i], point_tags[(i + 1) % number_of_vertices]
        )
        for i in range(number_of_vertices)
    ]
    loop = gmsh_module.model.occ.addCurveLoop(line_tags)
    surface = gmsh_module.model.occ.addPlaneSurface([loop])
    gmsh_module.model.occ.synchronize()

    gmsh_module.model.addPhysicalGroup(1, line_tags, name="boundary")
    gmsh_module.model.addPhysicalGroup(2, [surface], name="domain")

    if size_field_builder is not None:
        size_field_builder(gmsh_module)
        gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gmsh_module.model.mesh.generate(2)


def run_in_owned_gmsh_session(build: Callable[[Any], None]) -> Any:
    """Run ``build(gmsh_module)`` in a session this call owns, then extract.

    Initializes Gmsh only if no session is already open (so this composes
    with an outer caller -- e.g. ``create_gmsh_mesh_hierarchy`` -- that
    already owns a live session), and only finalizes what it itself opened.
    """
    gmsh_module = _import_gmsh()
    already_initialized = bool(gmsh_module.isInitialized())
    if not already_initialized:
        gmsh_module.initialize()
        gmsh_module.option.setNumber("General.Terminal", 0)
    try:
        build(gmsh_module)
        return create_triangle_mesh_from_gmsh_model(gmsh_module.model)
    finally:
        if not already_initialized:
            gmsh_module.finalize()


def create_gmsh_unit_square_mesh(number_of_cells_per_side: int) -> TriangleMesh:
    """
    Build a ``TriangleMesh`` of the unit square ``[0, 1]^2`` via Gmsh.

    Replaces the earlier structured (non-Gmsh) unit-square generator: Gmsh is
    the library's sole meshing engine for standard domains (Section 2.1).
    Boundary facets are tagged ``"bottom"`` (``y = 0``), ``"right"``
    (``x = 1``), ``"top"`` (``y = 1``), ``"left"`` (``x = 0``).

    Parameters
    ----------
    number_of_cells_per_side:
        Target number of cells along each side, at least 1.

    Returns
    -------
    TriangleMesh
    """
    return run_in_owned_gmsh_session(
        lambda gmsh_module: unit_square_gmsh_geometry(
            gmsh_module, number_of_cells_per_side
        )
    )


def create_gmsh_l_shaped_mesh(target_h: float) -> TriangleMesh:
    """
    Build a ``TriangleMesh`` of the re-entrant L-shaped domain via Gmsh.

    Domain: ``[-1, 1]^2 \\ [0, 1] x [-1, 0]``, with a genuine 270-degree
    re-entrant corner at the origin -- the standard benchmark domain for
    ``problem.create_poisson_l_shaped_singular_problem``. Every boundary
    facet is tagged ``"boundary"``.

    Parameters
    ----------
    target_h:
        Target mesh element size, greater than 0.

    Returns
    -------
    TriangleMesh
    """
    return run_in_owned_gmsh_session(
        lambda gmsh_module: l_shaped_gmsh_geometry(gmsh_module, target_h)
    )


def create_gmsh_mesh_hierarchy(
    geometry_builder: Callable[[Any, Any], None],
    base_resolution: Any,
    levels: int,
) -> list[TriangleMesh]:
    """
    Build a hierarchy of exact 1-to-4 uniformly refined ``TriangleMesh``\\ es.

    ``TriangleMesh`` is a pure, immutable data snapshot with no reference
    back to its originating CAD geometry, so refining an already-exported
    mesh directly is not possible. Instead, this function owns one live Gmsh
    session end to end: it runs ``geometry_builder(gmsh_module,
    base_resolution)`` once to build and mesh the coarsest level, exports it,
    then repeatedly calls ``gmsh.model.mesh.refine()`` (exact 1-to-4 red
    bisection, preserving physical group tags) and exports each subsequent
    level, before finalizing the session.

    Parameters
    ----------
    geometry_builder:
        A function ``(gmsh_module, base_resolution) -> None`` that builds a
        domain's geometry, tags its physical groups, and calls
        ``gmsh_module.model.mesh.generate(2)``, leaving the session open
        afterward -- e.g. ``unit_square_gmsh_geometry`` or
        ``l_shaped_gmsh_geometry``.
    base_resolution:
        The level-0 resolution argument passed to ``geometry_builder`` (an
        ``int`` cell count or a ``float`` target mesh size, depending on the
        builder).
    levels:
        Number of additional refinement levels beyond level 0, at least 0.
        Level ``i`` has exactly ``4**i`` times as many cells as level 0.

    Returns
    -------
    list[TriangleMesh]
        ``levels + 1`` meshes, coarsest (level 0) first.
    """
    if levels < 0:
        raise ValueError(f"levels must be non-negative, got {levels}.")

    gmsh_module = _import_gmsh()
    already_initialized = bool(gmsh_module.isInitialized())
    if not already_initialized:
        gmsh_module.initialize()
        gmsh_module.option.setNumber("General.Terminal", 0)
    try:
        geometry_builder(gmsh_module, base_resolution)
        meshes = [create_triangle_mesh_from_gmsh_model(gmsh_module.model)]
        for _ in range(levels):
            gmsh_module.model.mesh.refine()
            meshes.append(create_triangle_mesh_from_gmsh_model(gmsh_module.model))
        return meshes
    finally:
        if not already_initialized:
            gmsh_module.finalize()
