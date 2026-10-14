"""
Gmsh import of meshes and Gmsh-built standard domains.
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
    """
    Import the ``gmsh`` module.

    Returns
    -------
    gmsh : module
        The imported ``gmsh`` module.

    Raises
    ------
    ImportError
        If ``gmsh`` is not installed.
    """
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
    Build a triangular mesh from a meshed Gmsh model.

    Parameters
    ----------
    model : Any
        An object with the interface of ``gmsh.model`` holding a linear 2D
        mesh (e.g. after ``gmsh.model.mesh.generate(2)``).

    Returns
    -------
    mesh : TriangleMesh
        The mesh, independent of the live model.

    Notes
    -----
    Boundary facets are tagged from the model's 1D physical groups, matched
    exactly by canonical vertex-index pairs (no coordinate tolerance). Without
    1D physical groups every boundary facet gets the default tag. With 2D
    physical groups, each cell's tag is the Gmsh physical tag of its surface,
    or ``-1`` if untagged.
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
    Read a triangular mesh from a Gmsh ``.msh`` file.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Path to the ``.msh`` file.

    Returns
    -------
    mesh : TriangleMesh
        The mesh.
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
    Triangle elements of a Gmsh model.

    Parameters
    ----------
    model : Any
        The Gmsh model.

    Returns
    -------
    element_tags : np.ndarray
        Gmsh tag of each triangle, shape ``(K,)``.
    triangle_node_tags : np.ndarray
        Gmsh node tags of each triangle, shape ``(K, 3)``.
    """
    element_tags, node_tags_flat = model.mesh.getElementsByType(
        _TRIANGLE_ELEMENT_TYPE
    )
    element_tags = np.asarray(element_tags, dtype=np.int64)
    if element_tags.shape[0] == 0:
        raise ValueError(
            "The Gmsh model has no 3-node triangle elements. Call "
            "gmsh.model.mesh.generate(2) with a linear (order-1) mesh "
            "before importing."
        )
    triangle_node_tags = np.asarray(node_tags_flat, dtype=np.int64).reshape(-1, 3)
    return element_tags, triangle_node_tags


def _get_node_coordinates(model: Any) -> dict[int, tuple[float, float]]:
    """
    Node coordinates of a planar Gmsh model.

    Parameters
    ----------
    model : Any
        The Gmsh model.

    Returns
    -------
    tag_to_coords : dict[int, tuple[float, float]]
        Gmsh node tag to ``(x, y)`` coordinates.
    """
    node_tags, node_coords_flat, _ = model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coords = np.asarray(node_coords_flat, dtype=np.float64).reshape(-1, 3)

    if node_coords.shape[0] > 0 and not np.allclose(node_coords[:, 2], 0.0):
        raise ValueError(
            "The Gmsh model is not planar (found nonzero z coordinates). "
            "Only 2D straight-sided triangular meshes are supported."
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
    Boundary facet tags from the model's 1D physical curves.

    Parameters
    ----------
    model : Any
        The Gmsh model.
    node_tag_to_index : dict[int, int]
        Gmsh node tag to the mesh's vertex index.
    mesh : TriangleMesh
        The mesh whose boundary facets are tagged.

    Returns
    -------
    boundary_tags : tuple[np.ndarray, dict[str, int]] | None
        ``(boundary_facet_tags, boundary_tag_names)``, or ``None`` if the
        model has no 1D physical groups.

    Notes
    -----
    Line elements and mesh facets are matched by canonical vertex-index
    pairs built from the same node remap, so the lookup is exact.
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
    facets_to_vertices = np.asarray(mesh.facets_to_vertices)
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
    """
    Cell tags from the model's 2D physical surfaces.

    Parameters
    ----------
    model : Any
        The Gmsh model.
    element_tags : np.ndarray
        Gmsh tag of each triangle, shape ``(K,)``.

    Returns
    -------
    cell_tags : np.ndarray | None
        Physical tag of each cell, ``-1`` if untagged, shape ``(K,)``; or
        ``None`` if the model has no 2D physical groups.
    """
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


# Geometry builders mesh a domain in the current Gmsh session and leave it
# open, so they can be used directly or as the ``geometry_builder`` of
# ``create_gmsh_mesh_hierarchy``.


def unit_square_gmsh_geometry(gmsh_module: Any, number_of_cells_per_side: int) -> None:
    """
    Build and mesh the unit square in the current Gmsh session.

    Parameters
    ----------
    gmsh_module : Any
        The initialized ``gmsh`` module.
    number_of_cells_per_side : int
        Target number of cells along each side, at least one; the element
        size is ``1 / number_of_cells_per_side``.

    Notes
    -----
    Uses the ``geo`` kernel and tags the physical curves ``"bottom"``
    (``y = 0``), ``"right"`` (``x = 1``), ``"top"`` (``y = 1``), ``"left"``
    (``x = 0``) and the physical surface ``"domain"``.
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
    Build and mesh the L-shaped domain in the current Gmsh session.

    Parameters
    ----------
    gmsh_module : Any
        The initialized ``gmsh`` module.
    target_h : float | None
        Uniform target element size, positive; required unless
        ``size_field_builder`` is given.
    size_field_builder : Callable[[Any], None] | None
        Callback that installs a background mesh size field before meshing.

    Notes
    -----
    The domain is ``[-1, 1]^2 \\ [0, 1] x [-1, 0]`` with a 270-degree
    re-entrant corner at the origin. Uses the ``occ`` kernel and tags one
    physical curve ``"boundary"`` and the physical surface ``"domain"``.

    With ``size_field_builder``, CAD points get Gmsh's unconstrained size
    ``0.0`` and sizing from points and boundaries is disabled, so the field
    alone sets the element sizes; otherwise Gmsh would take the minimum of
    the point and field sizes, which can keep a coarse size at the corner.
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
    """
    Run a geometry builder in a Gmsh session and import the result.

    Parameters
    ----------
    build : Callable[[Any], None]
        Callback ``build(gmsh_module)`` that builds and meshes a domain.

    Returns
    -------
    mesh : TriangleMesh
        The imported mesh.

    Notes
    -----
    Gmsh is initialized only if no session is open, and only a session
    opened here is finalized, so the call composes with an outer session.
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
    Mesh of the unit square built with Gmsh.

    Parameters
    ----------
    number_of_cells_per_side : int
        Target number of cells along each side, at least one.

    Returns
    -------
    mesh : TriangleMesh
        The mesh, with boundary tags ``"bottom"``, ``"right"``, ``"top"``,
        and ``"left"``.
    """
    return run_in_owned_gmsh_session(
        lambda gmsh_module: unit_square_gmsh_geometry(
            gmsh_module, number_of_cells_per_side
        )
    )


def create_gmsh_l_shaped_mesh(target_h: float) -> TriangleMesh:
    """
    Mesh of the L-shaped domain built with Gmsh.

    Parameters
    ----------
    target_h : float
        Target element size, positive.

    Returns
    -------
    mesh : TriangleMesh
        The mesh of ``[-1, 1]^2 \\ [0, 1] x [-1, 0]``, with every boundary
        facet tagged ``"boundary"``.
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
    Uniformly refined mesh hierarchy built with Gmsh.

    Parameters
    ----------
    geometry_builder : Callable[[Any, Any], None]
        Callback ``(gmsh_module, base_resolution)`` that builds and meshes a
        domain and leaves the session open, e.g. ``unit_square_gmsh_geometry``
        or ``l_shaped_gmsh_geometry``.
    base_resolution : Any
        Level-0 resolution passed to ``geometry_builder`` (a cell count or an
        element size, depending on the builder).
    levels : int
        Number of refinement levels after level 0, at least zero.

    Returns
    -------
    meshes : list[TriangleMesh]
        ``levels + 1`` meshes, coarsest first; level ``i`` has ``4**i`` times
        as many cells as level 0.

    Notes
    -----
    A ``TriangleMesh`` keeps no link to its CAD geometry, so the hierarchy is
    built in one Gmsh session: the coarsest level is meshed once, then
    ``gmsh.model.mesh.refine()`` (exact 1-to-4 splitting that keeps physical
    groups) produces each further level.
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
