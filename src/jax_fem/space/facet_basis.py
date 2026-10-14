"""
Interior-facet geometry and gradients from both adjacent cells.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.reference_cell.quadrature import QuadratureRule
from jax_fem.space.cell_basis import CellBasis
from jax_fem.space.finite_element_space import FiniteElementSpace


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FacetBasis:
    """
    Geometry and two-sided basis gradients on every interior facet.

    Attributes
    ----------
    space : FiniteElementSpace
        The finite element space being evaluated.
    quadrature : QuadratureRule
        The facet quadrature rule, on a ``ReferenceInterval``.
    interior_facets : jax.Array
        Mesh facet index of each interior facet, shape ``(F,)``.
    cells_plus : jax.Array
        Cell on the plus side of each facet, shape ``(F,)``.
    cells_minus : jax.Array
        Cell on the minus side of each facet, shape ``(F,)``.
    physical_points : jax.Array
        Facet quadrature points in physical coordinates, shape
        ``(F, Q, 1, d)``.
    physical_weights : jax.Array
        Physical (arc-length) quadrature weights, shape ``(F, Q, 1, 1)``.
    facet_lengths : jax.Array
        Length of each facet, shape ``(F, 1, 1, 1)``.
    normals : jax.Array
        Unit normal of each facet, pointing from the plus to the minus cell,
        shape ``(F, 1, 1, d)``.
    gradients_plus : jax.Array
        Physical gradients of the plus cell's basis functions at the facet
        points, shape ``(F, Q, N_test, d)``.
    gradients_minus : jax.Array
        Physical gradients of the minus cell's basis functions at the facet
        points, shape ``(F, Q, N_test, d)``.

    Notes
    -----
    Only interior facets (two adjacent cells) are represented; boundary
    facets carry no jump term. The plus and minus sides follow the column
    order of ``mesh.facets_to_cells``, matching the jump convention
    ``J_e = (A grad u|+ - A grad u|-) . n_e``.
    """

    space: FiniteElementSpace
    quadrature: QuadratureRule
    interior_facets: jax.Array
    cells_plus: jax.Array
    cells_minus: jax.Array
    physical_points: jax.Array
    physical_weights: jax.Array
    facet_lengths: jax.Array
    normals: jax.Array
    gradients_plus: jax.Array
    gradients_minus: jax.Array

    def __post_init__(self) -> None:
        """
        Validate array shapes against the space and quadrature.

        Raises
        ------
        ValueError
            If a shape is inconsistent.
        """
        number_of_facets = self.interior_facets.shape[0]
        number_of_points = self.quadrature.points.shape[1]
        number_of_local_dofs = self.space.element.number_of_local_dofs
        dimension = self.space.mesh.geometric_dimension

        check_shape(self.interior_facets, (number_of_facets,), "interior_facets")
        check_shape(self.cells_plus, (number_of_facets,), "cells_plus")
        check_shape(self.cells_minus, (number_of_facets,), "cells_minus")
        check_shape(
            self.physical_points,
            (number_of_facets, number_of_points, 1, dimension),
            "physical_points",
        )
        check_shape(
            self.physical_weights,
            (number_of_facets, number_of_points, 1, 1),
            "physical_weights",
        )
        check_shape(self.facet_lengths, (number_of_facets, 1, 1, 1), "facet_lengths")
        check_shape(self.normals, (number_of_facets, 1, 1, dimension), "normals")
        check_shape(
            self.gradients_plus,
            (number_of_facets, number_of_points, number_of_local_dofs, dimension),
            "gradients_plus",
        )
        check_shape(
            self.gradients_minus,
            (number_of_facets, number_of_points, number_of_local_dofs, dimension),
            "gradients_minus",
        )


def create_facet_basis(basis: CellBasis, quadrature: QuadratureRule) -> FacetBasis:
    """
    Evaluate interior-facet geometry and two-sided basis gradients.

    Parameters
    ----------
    basis : CellBasis
        Evaluated cell basis of the space; only its inverse Jacobians are
        used.
    quadrature : QuadratureRule
        Quadrature rule on a ``ReferenceInterval``.

    Returns
    -------
    facet_basis : FacetBasis
        The interior-facet geometry and gradients.

    Raises
    ------
    NotImplementedError
        If the mesh is embedded in a higher-dimensional space.

    Notes
    -----
    The normal is the facet tangent rotated by 90 degrees, with the sign
    that points from the plus cell's centroid towards the minus cell's. The
    gradients on each side are obtained by mapping the physical facet points
    back into that side's reference triangle.
    """
    space = basis.space
    mesh = space.mesh
    element = space.element

    if mesh.geometric_dimension != mesh.topological_dimension:
        raise NotImplementedError(
            "create_facet_basis supports planar meshes only; facets of an "
            "embedded mesh can have more than two cells and no in-plane normal."
        )

    interior_mask = jnp.all(mesh.facets_to_cells != -1, axis=-1)
    interior_facets = jnp.nonzero(interior_mask)[0].astype(jnp.int32)
    cells_plus = mesh.facets_to_cells[interior_facets, 0]
    cells_minus = mesh.facets_to_cells[interior_facets, 1]

    facet_vertices = mesh.facets_to_vertices[interior_facets]
    first_vertices = jnp.expand_dims(
        mesh.vertex_coordinates[facet_vertices[..., 0]], axis=-3
    )
    second_vertices = jnp.expand_dims(
        mesh.vertex_coordinates[facet_vertices[..., 1]], axis=-3
    )
    edge_vectors = second_vertices - first_vertices
    facet_lengths = jnp.linalg.norm(edge_vectors, axis=-1, keepdims=True)

    physical_points = first_vertices + quadrature.points * edge_vectors
    physical_weights = (
        quadrature.weights * quadrature.reference_cell.measure * facet_lengths
    )

    tangents = edge_vectors / facet_lengths
    perpendiculars = jnp.stack((-tangents[..., 1], tangents[..., 0]), axis=-1)
    cell_centroids = jnp.expand_dims(
        jnp.mean(mesh.vertex_coordinates[mesh.cells_to_vertices], axis=-3), axis=-3
    )
    directions = cell_centroids[cells_minus] - cell_centroids[cells_plus]
    signs = jnp.sign(jnp.sum(perpendiculars * directions, axis=-1, keepdims=True))
    normals = signs * perpendiculars

    origins = jnp.expand_dims(
        mesh.vertex_coordinates[mesh.cells_to_vertices[..., 0]], axis=-3
    )

    def side_gradients(cells_side: jax.Array) -> jax.Array:
        """
        Physical basis gradients of one side's cells at the facet points.

        Parameters
        ----------
        cells_side : jax.Array
            Cell on this side of each facet, shape ``(F,)``.

        Returns
        -------
        gradients : jax.Array
            Physical gradients, shape ``(F, Q, N_test, d)``.
        """
        inverse_jacobians_side = basis.inverse_jacobians[cells_side]
        displacements = physical_points - origins[cells_side]
        reference_points_side = displacements @ inverse_jacobians_side.mT
        reference_gradients_side = element.tabulate_basis_gradients(
            reference_points_side
        )
        return jnp.broadcast_to(
            reference_gradients_side @ inverse_jacobians_side,
            physical_points.shape[:-2]
            + (element.number_of_local_dofs, mesh.geometric_dimension),
        )

    return FacetBasis(
        space=space,
        quadrature=quadrature,
        interior_facets=interior_facets,
        cells_plus=cells_plus,
        cells_minus=cells_minus,
        physical_points=physical_points,
        physical_weights=physical_weights,
        facet_lengths=facet_lengths,
        normals=normals,
        gradients_plus=side_gradients(cells_plus),
        gradients_minus=side_gradients(cells_minus),
    )
