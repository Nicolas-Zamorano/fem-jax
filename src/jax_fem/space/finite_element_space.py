"""
Finite element spaces and global DOF numbering.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.element.base import FiniteElement
from jax_fem.mesh.base import Mesh

#: Mesh-entity dimensions that ``entity_dofs`` may assign DOFs to:
#: 0 vertices, 1 facets, 2 cell interiors.
_ENTITY_DIMENSIONS = (0, 1, 2)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FiniteElementSpace:
    """
    A global finite element space built from a mesh and an element.

    Attributes
    ----------
    mesh : Mesh
        The mesh the space is built on.
    element : FiniteElement
        The local finite element.
    cells_to_dofs : jax.Array
        Global DOF index of each local DOF, shape ``(K, N_test)``.
    dof_coordinates : jax.Array
        Coordinate associated with each global DOF, shape
        ``(dofs_test, 1, g)`` with ``g = mesh.geometric_dimension``.
    number_of_dofs : int
        Number of global DOFs.
    entity_dof_offset : Mapping[int, int]
        First global DOF index of each entity dimension that carries DOFs.
    local_dof_signs : jax.Array
        Sign (``+1`` or ``-1``) of each local DOF, shape ``(K, N_test)``.

    Notes
    -----
    Entity ``e`` of dimension ``d`` owns global DOF
    ``e + entity_dof_offset[d]``; this holds because every element assigns at
    most one DOF per entity, and ``find_boundary_dofs`` relies on it.

    ``local_dof_signs`` is all ``+1`` unless the element is orientation
    dependent (RT0). Then a facet DOF gets ``+1`` when the cell's local facet
    vertex order matches the mesh's canonical ``(low, high)`` order and
    ``-1`` otherwise, so a shared facet's global DOF has one meaning from
    both sides. ``create_cell_basis`` multiplies the signs into the physical
    values and divergences.
    """

    mesh: Mesh
    element: FiniteElement
    cells_to_dofs: jax.Array
    dof_coordinates: jax.Array
    number_of_dofs: int
    entity_dof_offset: Mapping[int, int]
    local_dof_signs: jax.Array

    def __post_init__(self) -> None:
        """
        Validate the shapes of the DOF arrays.

        Raises
        ------
        ValueError
            If an array does not match the mesh, the element, or
            ``number_of_dofs``.
        """
        number_of_cells = self.mesh.cells_to_vertices.shape[0]
        check_shape(
            self.cells_to_dofs,
            (number_of_cells, self.element.number_of_local_dofs),
            "cells_to_dofs",
        )
        check_shape(
            self.dof_coordinates,
            (self.number_of_dofs, 1, self.mesh.geometric_dimension),
            "dof_coordinates",
        )
        check_shape(
            self.local_dof_signs,
            (number_of_cells, self.element.number_of_local_dofs),
            "local_dof_signs",
        )


def create_finite_element_space(
    mesh: Mesh, element: FiniteElement
) -> FiniteElementSpace:
    """
    Build a finite element space from a mesh and an element.

    Parameters
    ----------
    mesh : Mesh
        The mesh to build the space on.
    element : FiniteElement
        The local finite element.

    Returns
    -------
    space : FiniteElementSpace
        The global space.

    Notes
    -----
    Numbering follows ``element.entity_dofs``: for each entity dimension the
    element uses (in the order 0, 1, 2), that dimension's mesh entities are
    numbered consecutively from the running offset, and each local DOF is
    mapped to its entity's global index. Dimensions without DOFs are
    skipped. This gives vertex numbering for P1, vertices then facets for
    P2, and cells for P0, with no element-specific code.
    """
    if not isinstance(element, FiniteElement):
        raise TypeError(
            "element must implement the FiniteElement protocol: "
            "reference_cell, polynomial_degree, number_of_local_dofs, "
            "value_shape, entity_dofs, is_nodal, tabulate_basis_values, and "
            "tabulate_basis_gradients."
        )

    number_of_cells = int(mesh.cells_to_vertices.shape[0])
    cells_to_dofs = jnp.zeros(
        (number_of_cells, element.number_of_local_dofs), dtype=jnp.int32
    )
    local_dof_signs = jnp.ones((number_of_cells, element.number_of_local_dofs))
    dof_coordinate_blocks: list[jax.Array] = []
    entity_dof_offset: dict[int, int] = {}
    offset = 0

    facet_signs = (
        _local_facet_signs(mesh, element.reference_cell)
        if element.is_orientation_dependent
        else None
    )

    for dimension in _ENTITY_DIMENSIONS:
        local_dofs_per_entity = element.entity_dofs.get(dimension, ())
        if not any(local_dofs_per_entity):
            continue

        entity_dof_offset[dimension] = offset
        cells_to_entities, number_of_entities = _mesh_entity_indices_and_count(
            mesh, dimension
        )
        dof_coordinate_blocks.append(
            _entity_representative_coordinates(mesh, dimension)
        )

        for local_entity, local_dof_ids in enumerate(local_dofs_per_entity):
            if not local_dof_ids:
                continue
            global_entity_dofs = (
                cells_to_entities[:, local_entity].astype(jnp.int32) + offset
            )
            for local_dof_id in local_dof_ids:
                cells_to_dofs = cells_to_dofs.at[:, local_dof_id].set(
                    global_entity_dofs
                )
                if dimension == 1 and facet_signs is not None:
                    local_dof_signs = local_dof_signs.at[:, local_dof_id].set(
                        facet_signs[:, local_entity]
                    )

        offset += number_of_entities

    number_of_dofs = offset
    dof_coordinates = (
        jnp.concatenate(dof_coordinate_blocks, axis=0)
        if dof_coordinate_blocks
        else jnp.zeros((0, 1, mesh.geometric_dimension))
    )

    if number_of_dofs > 0 and int(jnp.max(cells_to_dofs)) >= number_of_dofs:
        raise ValueError("cells_to_dofs references a DOF index >= number_of_dofs.")

    return FiniteElementSpace(
        mesh=mesh,
        element=element,
        cells_to_dofs=cells_to_dofs,
        dof_coordinates=dof_coordinates,
        number_of_dofs=number_of_dofs,
        entity_dof_offset=entity_dof_offset,
        local_dof_signs=local_dof_signs,
    )


def _local_facet_signs(mesh: Mesh, reference_cell) -> jax.Array:
    """
    Orientation sign of every local facet.

    Parameters
    ----------
    mesh : Mesh
        The mesh; its ``facets_to_vertices`` are sorted ``(low, high)``.
    reference_cell : ReferenceCell
        Reference cell giving the local facet vertex order.

    Returns
    -------
    signs : jax.Array
        ``+1`` where the cell's local facet vertex order is increasing (the
        mesh's canonical order) and ``-1`` where it is reversed, shape
        ``(K, 3)``.
    """
    local_facet_vertices = reference_cell.facets_to_vertices
    cell_facet_vertex_ids = mesh.cells_to_vertices[:, local_facet_vertices]
    ascending = cell_facet_vertex_ids[..., 0] < cell_facet_vertex_ids[..., 1]
    return jnp.where(ascending, 1.0, -1.0)


def _mesh_entity_indices_and_count(mesh: Mesh, dimension: int) -> tuple[jax.Array, int]:
    """
    Global entity index of each local entity, and the number of entities.

    Parameters
    ----------
    mesh : Mesh
        The mesh; dimension 1 needs its ``cells_to_facets`` and
        ``facets_to_vertices``.
    dimension : int
        Entity dimension: ``0`` vertices, ``1`` facets, ``2`` cell interiors.

    Returns
    -------
    cells_to_entities : jax.Array
        Global entity index of each local entity, shape ``(K, 3)`` for
        dimensions 0 and 1 and ``(K, 1)`` for dimension 2.
    number_of_entities : int
        Number of mesh entities of this dimension.
    """
    if dimension == 0:
        return mesh.cells_to_vertices, int(mesh.vertex_coordinates.shape[0])
    if dimension == 1:
        return mesh.cells_to_facets, int(mesh.facets_to_vertices.shape[0])
    if dimension == 2:
        number_of_cells = int(mesh.cells_to_vertices.shape[0])
        cell_indices = jnp.arange(number_of_cells, dtype=jnp.int32)
        return jnp.expand_dims(cell_indices, axis=-1), number_of_cells
    raise ValueError(
        f"Unsupported entity_dofs dimension {dimension}; expected 0, 1, or 2."
    )


def _entity_representative_coordinates(mesh: Mesh, dimension: int) -> jax.Array:
    """
    One representative coordinate per mesh entity.

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    dimension : int
        Entity dimension: ``0`` vertices, ``1`` facets, ``2`` cell interiors.

    Returns
    -------
    coordinates : jax.Array
        Vertex positions, facet midpoints, or cell centroids, shape
        ``(n_entities, 1, g)``.
    """
    if dimension == 0:
        return mesh.vertex_coordinates
    if dimension == 1:
        facet_vertex_coordinates = mesh.vertex_coordinates[mesh.facets_to_vertices]
        return jnp.mean(facet_vertex_coordinates, axis=-3)
    if dimension == 2:
        cell_vertex_coordinates = mesh.vertex_coordinates[mesh.cells_to_vertices]
        return jnp.mean(cell_vertex_coordinates, axis=-3)
    raise ValueError(
        f"Unsupported entity_dofs dimension {dimension}; expected 0, 1, or 2."
    )


def find_boundary_dofs(
    space: FiniteElementSpace, boundary_tags: tuple[int, ...]
) -> jax.Array:
    """
    Global DOF indices on the given boundary tags.

    Parameters
    ----------
    space : FiniteElementSpace
        The finite element space.
    boundary_tags : tuple[int, ...]
        Mesh boundary facet tags whose DOFs are returned.

    Returns
    -------
    boundary_dofs : jax.Array
        Sorted, unique global DOF indices, shape ``(N_selected,)``.

    Notes
    -----
    Collects the DOFs owned by the endpoint vertices of the selected facets
    (dimension 0) and by the facets themselves (dimension 1). An element with
    DOFs on neither, such as P0, has no boundary DOFs and gets an empty
    array.
    """
    mesh = space.mesh
    selected_tags = jnp.asarray(boundary_tags)
    selected_facets_mask = jnp.isin(mesh.boundary_facet_tags, selected_tags)
    selected_facets = mesh.boundary_facets[selected_facets_mask]

    contributions: list[jax.Array] = []

    if 0 in space.entity_dof_offset:
        selected_facet_vertices = mesh.facets_to_vertices[selected_facets]
        contributions.append(
            jnp.ravel(selected_facet_vertices + space.entity_dof_offset[0])
        )

    if 1 in space.entity_dof_offset:
        contributions.append(selected_facets + space.entity_dof_offset[1])

    if not contributions:
        return jnp.zeros((0,), dtype=jnp.int32)

    return jnp.unique(jnp.concatenate(contributions))
