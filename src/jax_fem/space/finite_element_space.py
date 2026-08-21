"""Finite-element spaces and global DOF numbering.

See Section 10.1 of the architecture specification, and Section 3.2/3.3 of
the implementation plan (``.context/implementation_plan.md``) for the
generic ``entity_dofs``-driven DOF numbering used here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.element.base import FiniteElement
from jax_fem.mesh.base import Mesh

if TYPE_CHECKING:
    from jax_fem.function.finite_element_function import FiniteElementFunction

#: Mesh-entity dimensions a FiniteElement's entity_dofs may assign DOFs to:
#: 0 = vertices, 1 = facets/edges, 2 = cell interior.
_ENTITY_DIMENSIONS = (0, 1, 2)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FiniteElementSpace:
    """A mesh-element pair combined into a global discrete space ``V_h``.

    Attributes
    ----------
    mesh : Mesh
        The mesh to build the space on.
    element : FiniteElement
        The local finite element.
    cells_to_dofs : jax.Array
        Global DOF index of each cell's local DOFs, shape ``(K, N_phi)``.
    dof_coordinates : jax.Array
        Coordinate associated with each global DOF, shape ``(N_dofs, 1, 2)``.
    number_of_dofs : int
        The number of global DOFs.
    entity_dof_offset : Mapping[int, int]
        For each mesh-entity dimension the element owns at least one DOF on
        (a subset of ``{0, 1, 2}``), the global DOF index at which that
        dimension's entities begin being numbered: entity ``e`` of
        dimension ``d`` (``d in entity_dof_offset``) owns global DOF index
        ``e + entity_dof_offset[d]``. Valid because every element in this
        library assigns at most one DOF per mesh entity (``FiniteElement
        .entity_dofs``); consumed generically by ``find_boundary_dofs``.
    local_dof_signs : jax.Array
        Sign (``+1`` or ``-1``) each cell's local DOF must be multiplied by
        before use, shape ``(K, N_phi)``. All ``+1`` unless
        ``element.is_orientation_dependent`` (e.g. RT0): a facet-owned
        flux DOF's sign depends on whether the cell's local facet vertex
        ordering matches the mesh's global canonical ``(low, high)``
        ordering (Section 5.2's ``S`` matrix) -- +1 if it does, -1 if
        reversed, so a shared facet's global DOF has one unambiguous
        physical meaning regardless of which adjacent cell computed it.
        Folded into physical values/divergences by ``create_cell_basis``.
    """

    mesh: Mesh
    element: FiniteElement
    cells_to_dofs: jax.Array
    dof_coordinates: jax.Array
    number_of_dofs: int
    entity_dof_offset: Mapping[int, int]
    local_dof_signs: jax.Array

    def __post_init__(self) -> None:
        number_of_cells = self.mesh.cells_to_vertices.shape[0]
        check_shape(
            self.cells_to_dofs,
            (number_of_cells, self.element.number_of_local_dofs),
            "cells_to_dofs",
        )
        check_shape(
            self.dof_coordinates, (self.number_of_dofs, 1, 2), "dof_coordinates"
        )
        check_shape(
            self.local_dof_signs,
            (number_of_cells, self.element.number_of_local_dofs),
            "local_dof_signs",
        )


def create_finite_element_space(
    mesh: Mesh, element: FiniteElement
) -> FiniteElementSpace:
    """Combine a mesh and finite element into a ``FiniteElementSpace``.

    Global DOF numbering is driven generically by ``element.entity_dofs``:
    for each mesh-entity dimension the element uses (in order 0, 1, 2), that
    dimension's mesh entities are numbered consecutively starting at the
    running offset, and every local DOF owned by a local entity of that
    dimension is mapped, per cell, to its entity's global index. Dimensions
    the element assigns no DOFs to are skipped entirely (contributing no
    DOFs and no offset). This reproduces plain vertex numbering for
    ``LagrangeTriangleP1`` and vertex-then-facet numbering for
    ``LagrangeTriangleP2`` without any element-specific branching.

    Parameters
    ----------
    mesh: Mesh
        The mesh to build the space on.
    element: FiniteElement
        The local finite element.

    Returns
    -------
    FiniteElementSpace
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
            continue  # this element assigns no DOFs to entities of this dimension

        entity_dof_offset[dimension] = offset
        cells_to_entities, number_of_entities = _mesh_entity_indices_and_count(
            mesh, dimension
        )
        dof_coordinate_blocks.append(_entity_representative_coordinates(mesh, dimension))

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
        else jnp.zeros((0, 1, 2))
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
    """Per-cell, per-local-facet orientation sign, shape ``(K, 3)``.

    ``+1`` if the cell's local facet vertex ordering
    (``reference_cell.facets_to_vertices``, applied through
    ``mesh.cells_to_vertices``) already increases -- matching the mesh's
    global canonical ``(low, high)`` facet vertex ordering
    (``mesh.facets_to_vertices``, always sorted ascending by construction)
    -- ``-1`` if it is reversed.
    """
    local_facet_vertices = reference_cell.facets_to_vertices  # (3, 2)
    cell_facet_vertex_ids = mesh.cells_to_vertices[:, local_facet_vertices]  # (K,3,2)
    ascending = cell_facet_vertex_ids[..., 0] < cell_facet_vertex_ids[..., 1]
    return jnp.where(ascending, 1.0, -1.0)


def _mesh_entity_indices_and_count(mesh: Mesh, dimension: int) -> tuple[jax.Array, int]:
    """Local-entity -> global-entity index array, and total entity count.

    Parameters
    ----------
    mesh: Mesh
        The mesh (in practice always a ``TriangleMesh``, whose facet
        topology this relies on for ``dimension in (1, 2)``; the ``Mesh``
        protocol itself only guarantees ``dimension == 0``).
    dimension: int
        Entity dimension: ``0`` (vertices), ``1`` (facets), or ``2`` (cell
        interior).

    Returns
    -------
    cells_to_entities : jax.Array
        Shape ``(K, n_local_entities_of_dimension)``: 3 for dimension 0 or
        1, 1 for dimension 2.
    number_of_entities : int
        Total mesh entities of this dimension.
    """
    if dimension == 0:
        return mesh.cells_to_vertices, int(mesh.vertex_coordinates.shape[0])
    if dimension == 1:
        return mesh.cells_to_facets, int(mesh.facets_to_vertices.shape[0])
    if dimension == 2:
        number_of_cells = int(mesh.cells_to_vertices.shape[0])
        cell_indices = jnp.arange(number_of_cells, dtype=jnp.int32)[:, None]
        return cell_indices, number_of_cells
    raise ValueError(f"Unsupported entity_dofs dimension {dimension}; expected 0, 1, or 2.")


def _entity_representative_coordinates(mesh: Mesh, dimension: int) -> jax.Array:
    """One representative coordinate per mesh entity of ``dimension``.

    Vertex position, facet midpoint, or cell centroid, shape
    ``(n_entities_of_dimension, 1, 2)``.
    """
    if dimension == 0:
        return mesh.vertex_coordinates
    if dimension == 1:
        facet_vertex_coordinates = mesh.vertex_coordinates[mesh.facets_to_vertices]
        return jnp.mean(facet_vertex_coordinates, axis=1)
    if dimension == 2:
        cell_vertex_coordinates = mesh.vertex_coordinates[mesh.cells_to_vertices]
        return jnp.mean(cell_vertex_coordinates, axis=1)
    raise ValueError(f"Unsupported entity_dofs dimension {dimension}; expected 0, 1, or 2.")


def find_boundary_dofs(
    space: FiniteElementSpace, boundary_tags: tuple[int, ...]
) -> jax.Array:
    """Return the global DOF indices on the given mesh boundary tags.

    Generic over any element's ``entity_dofs``: unions DOFs owned by
    selected boundary facets' endpoint vertices (dimension 0) with DOFs
    owned by the facets themselves (dimension 1). An element with DOFs on
    neither dimension (e.g. a purely cell-interior element) has no boundary
    DOFs by this definition, and returns an empty array.

    Parameters
    ----------
    space : FiniteElementSpace
        The finite element space to query.
    boundary_tags : tuple[int, ...]
        Mesh boundary facet tags whose DOFs should be returned.

    Returns
    -------
    boundary_dofs : jax.Array
        Sorted, deduplicated global DOF indices, shape ``(N_selected,)``.
    """
    mesh = space.mesh
    selected_tags = jnp.asarray(boundary_tags)
    selected_facets_mask = jnp.isin(mesh.boundary_facet_tags, selected_tags)
    selected_facets = mesh.boundary_facets[selected_facets_mask]

    contributions: list[jax.Array] = []

    if 0 in space.entity_dof_offset:
        selected_facet_vertices = mesh.facets_to_vertices[selected_facets]  # (Ns, 2)
        contributions.append(
            (selected_facet_vertices + space.entity_dof_offset[0]).reshape(-1)
        )

    if 1 in space.entity_dof_offset:
        contributions.append(selected_facets + space.entity_dof_offset[1])

    if not contributions:
        return jnp.zeros((0,), dtype=jnp.int32)

    return jnp.unique(jnp.concatenate(contributions))


def gather_cell_dof_values(function: FiniteElementFunction) -> jax.Array:
    """
    Gather a finite element function's global DOF values onto each cell.

    Parameters
    ----------
    function : FiniteElementFunction
        The finite element function to gather.

    Returns
    -------
    local_values : jax.Array
        Local DOF values, shape ``(K, N_phi, 1)``.
    """
    return function.dof_values[function.space.cells_to_dofs]
