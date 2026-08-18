"""
Gmsh model and ``.msh`` file import paths.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_fem.mesh.triangle import (
    BoundaryData,
    TriangleMesh,
    create_triangle_mesh_from_arrays,
)

# Gmsh's fixed elementary type codes (stable across Gmsh versions):
# https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
_TRIANGLE_ELEMENT_TYPE = 2  # 3-node triangle
_LINE_ELEMENT_TYPE = 1  # 2-node line

# Coincidence tolerance when matching a boundary facet midpoint (derived from
# our own vertex_coordinates) against a Gmsh physical-curve edge midpoint
# (derived independently from the same underlying node coordinates).
_MIDPOINT_MATCH_TOLERANCE = 1e-9


def create_triangle_mesh_from_gmsh_model(model: Any) -> TriangleMesh:
    """
    Build a ``TriangleMesh`` from a live, already-meshed Gmsh model.

    The model must already have a linear (order-1) 2D mesh generated, e.g.
    via ``gmsh.model.mesh.generate(2)``. Boundary facets are tagged from the
    model's 1D physical groups (physical curves); a facet's midpoint is
    matched against the midpoints of that physical group's line elements. If
    the model has no 1D physical groups, every boundary facet receives the
    single default tag (Section 7.3). If the model has 2D physical groups
    (physical surfaces), each cell's ``cell_tags`` entry is set to the raw
    Gmsh physical tag of the surface it belongs to, or ``-1`` if untagged.

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
    #FIXME: this should not use boundary tags of gmsh?
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

    boundary_data = _boundary_predicates_from_physical_curves(model, tag_to_coords)

    mesh = create_triangle_mesh_from_arrays(
        vertex_coordinates, cells_to_vertices, boundary_data=boundary_data
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
    try:
        import gmsh
    except ImportError as error:
        raise ImportError(
            "read_triangle_mesh_from_msh requires the optional 'gmsh' "
            "dependency. Install it with: pip install 'jax-fem[gmsh]'."
        ) from error

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


def _boundary_predicates_from_physical_curves(
    model: Any, tag_to_coords: dict[int, tuple[float, float]]
) -> BoundaryData | None:
    physical_groups = model.getPhysicalGroups(dim=1)
    if not physical_groups:
        return None

    boundary_data: dict[str, Callable[[jax.Array], jax.Array]] = {}
    for dim, physical_tag in physical_groups:
        name = model.getPhysicalName(dim, physical_tag) or f"boundary_{physical_tag}"
        curve_tags = model.getEntitiesForPhysicalGroup(dim, physical_tag)

        edge_midpoints: list[tuple[float, float]] = []
        for curve_tag in curve_tags:
            _, line_node_tags_flat = model.mesh.getElementsByType(
                _LINE_ELEMENT_TYPE, tag=curve_tag
            )
            line_node_tags = np.asarray(line_node_tags_flat, dtype=np.int64).reshape(
                -1, 2
            )
            for node_a, node_b in line_node_tags:
                coord_a = tag_to_coords[int(node_a)]
                coord_b = tag_to_coords[int(node_b)]
                edge_midpoints.append(
                    (
                        (coord_a[0] + coord_b[0]) / 2.0,
                        (coord_a[1] + coord_b[1]) / 2.0,
                    )
                )

        boundary_data[name] = _make_midpoint_membership_predicate(edge_midpoints)

    return boundary_data


def _make_midpoint_membership_predicate(
    known_midpoints: list[tuple[float, float]],
) -> Callable[[jax.Array], jax.Array]:
    known_midpoints_array = jnp.asarray(known_midpoints)  # (M, 2)

    def predicate(facet_midpoints: jax.Array) -> jax.Array:
        # facet_midpoints: (Nb, 1, 2)
        differences = facet_midpoints[:, 0, None, :] - known_midpoints_array[None]
        distances = jnp.linalg.norm(differences, axis=-1)  # (Nb, M)
        return jnp.any(distances < _MIDPOINT_MATCH_TOLERANCE, axis=-1)

    return predicate


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
