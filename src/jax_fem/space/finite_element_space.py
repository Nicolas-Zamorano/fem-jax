"""Finite-element spaces and global DOF numbering.

See Section 10.1 of the architecture specification.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.element.base import FiniteElement
from jax_fem.element.lagrange_triangle import LagrangeTriangleP1
from jax_fem.mesh.base import Mesh

if TYPE_CHECKING:
    from jax_fem.function.finite_element_function import FiniteElementFunction


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
    """

    mesh: Mesh
    element: FiniteElement
    cells_to_dofs: jax.Array
    dof_coordinates: jax.Array
    number_of_dofs: int

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


def create_finite_element_space(
    mesh: Mesh, element: FiniteElement
) -> FiniteElementSpace:
    """Combine a mesh and finite element into a ``FiniteElementSpace``.

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
    if not isinstance(element, LagrangeTriangleP1):
        raise NotImplementedError(
            "create_finite_element_space currently only supports "
            "LagrangeTriangleP1. DOF numbering for other elements (e.g. "
            "higher-order Lagrange elements with entity-owned edge/interior "
            "DOFs) is a future extension (Section 18.1)."
        )

    # DOF numbering for P1 elements on a triangular mesh: DOFs coincide
    # 1:1 with mesh vertices.
    cells_to_dofs = jnp.asarray(mesh.cells_to_vertices, dtype=jnp.int32)
    dof_coordinates = jnp.asarray(mesh.vertex_coordinates)
    number_of_dofs = int(mesh.vertex_coordinates.shape[0])

    if number_of_dofs > 0 and int(jnp.max(cells_to_dofs)) >= number_of_dofs:
        raise ValueError("cells_to_dofs references a DOF index >= number_of_dofs.")

    return FiniteElementSpace(
        mesh=mesh,
        element=element,
        cells_to_dofs=cells_to_dofs,
        dof_coordinates=dof_coordinates,
        number_of_dofs=number_of_dofs,
    )


def find_boundary_dofs(
    space: FiniteElementSpace, boundary_tags: tuple[int, ...]
) -> jax.Array:
    """Return the global DOF indices on the given mesh boundary tags.

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
    selected_facet_vertices = mesh.facets_to_vertices[selected_facets]  # (Ns, 2)
    return jnp.unique(selected_facet_vertices)


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
