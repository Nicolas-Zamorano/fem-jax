"""
Interior-facet geometry and both-sided physical gradient evaluation.

See Section 4.2 of the implementation plan (``.context/implementation_plan.md``).
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
    Geometry and both-sided physical gradients at every interior mesh facet.

    Only *interior* facets (shared by exactly two cells) are represented:
    boundary facets carry no normal-jump term in the residual error
    estimator (Section 4.1) and are excluded entirely up front, rather than
    padded with a sentinel cell.

    Each interior facet has a "plus" and "minus" side (``cells_plus`` /
    ``cells_minus``, from ``mesh.facets_to_cells``' own column order); the
    unit normal points from the plus side to the minus side, matching the
    jump convention ``J_e = (A grad(u_h)|+ - A grad(u_h)|-) . n_e``.

    Attributes
    ----------
    space : FiniteElementSpace
        The finite element space being evaluated.
    quadrature : QuadratureRule
        The 1D quadrature rule (on a ``ReferenceInterval``) facets are
        evaluated at.
    interior_facets : jax.Array
        Global mesh facet index of each represented facet, shape ``(F,)``.
    cells_plus : jax.Array
        Global cell index of each facet's plus side, shape ``(F,)``.
    cells_minus : jax.Array
        Global cell index of each facet's minus side, shape ``(F,)``.
    physical_points : jax.Array
        Quadrature points mapped to physical coordinates, shape
        ``(F, Q, 1, d)``.
    physical_weights : jax.Array
        Physical (arc-length) integration weights, shape ``(F, Q, 1, 1)``.
    facet_lengths : jax.Array
        Physical length of each facet, shape ``(F, 1, 1, 1)``.
    normals : jax.Array
        Unit normal of each facet, pointing from the plus side to the minus
        side, constant along the facet, shape ``(F, 1, 1, d)``.
    gradients_plus : jax.Array
        Physical gradients of ``space.element``'s local basis functions,
        evaluated at ``physical_points`` from ``cells_plus``'s own affine
        map (i.e. what each plus-side basis function's gradient equals at
        that physical point), shape ``(F, Q, N_phi, d)``.
    gradients_minus : jax.Array
        As ``gradients_plus``, from ``cells_minus``, shape
        ``(F, Q, N_phi, d)``.
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
    Evaluate interior-facet geometry and both-sided physical gradients.

    Parameters
    ----------
    basis : CellBasis
        The space's cell geometry, already evaluated at some 2D quadrature
        rule; only ``basis.inverse_jacobians`` (a purely per-cell, 2D-
        quadrature-independent quantity) is actually used here.
    quadrature : QuadratureRule
        A 1D quadrature rule on ``ReferenceInterval`` (Section 4.2).

    Returns
    -------
    FacetBasis
    """
    space = basis.space
    mesh = space.mesh
    element = space.element

    interior_mask = jnp.all(mesh.facets_to_cells != -1, axis=1)
    interior_facets = jnp.nonzero(interior_mask)[0].astype(jnp.int32)

    cells_plus = mesh.facets_to_cells[interior_facets, 0]
    cells_minus = mesh.facets_to_cells[interior_facets, 1]

    facet_vertices = mesh.facets_to_vertices[interior_facets]  # (F, 2)
    vertex_a = mesh.vertex_coordinates[facet_vertices[:, 0]]  # (F, 1, 2)
    vertex_b = mesh.vertex_coordinates[facet_vertices[:, 1]]  # (F, 1, 2)
    edge_vector = vertex_b - vertex_a  # (F, 1, 2)
    facet_lengths = jnp.linalg.norm(edge_vector, axis=-1).reshape(-1, 1, 1, 1)

    # t: the 1D reference-interval quadrature parameter, shape (1, Q, 1, 1);
    # its trailing size-1 axis broadcasts against the (F, 1, 1, 2) physical
    # edge vector directly (a scalar-per-point multiplier of a 2D vector).
    t = quadrature.points
    physical_points = vertex_a[:, None, :, :] + t * edge_vector[:, None, :, :]
    physical_weights = quadrature.weights * quadrature.reference_cell.measure * (
        facet_lengths
    )

    # Unit normal, plus -> minus: rotate the tangent 90 degrees, then pick
    # the sign that points from the plus cell's centroid toward the minus
    # cell's.
    tangent = edge_vector / facet_lengths[:, 0]  # (F, 1, 2)
    perpendicular = jnp.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)  # (F, 1, 2)
    cell_centroids = jnp.mean(
        mesh.vertex_coordinates[mesh.cells_to_vertices], axis=1
    )  # (K, 1, 2)
    direction = cell_centroids[cells_minus] - cell_centroids[cells_plus]  # (F, 1, 2)
    sign = jnp.sign(jnp.sum(perpendicular * direction, axis=-1, keepdims=True))
    normals = (sign * perpendicular)[:, None, :, :]  # (F, 1, 1, 2)

    origin = mesh.vertex_coordinates[mesh.cells_to_vertices[:, 0]][:, None, :, :]  # (K,1,1,2)

    def side_gradients(cells_side: jax.Array) -> jax.Array:
        origin_side = origin[cells_side]  # (F, 1, 1, 2)
        inverse_jacobian_side = basis.inverse_jacobians[cells_side]  # (F, 1, 2, 2)
        displacement = physical_points - origin_side  # (F, Q, 1, 2)
        # Inverse affine map, row-vector convention (create_cell_basis's
        # forward map is x_row = origin_row + xhat_row @ J^T):
        # xhat_row = (x - origin)_row @ (J^{-1})^T.
        reference_points_side = displacement @ inverse_jacobian_side.mT  # (F, Q, 1, 2)
        reference_gradients_side = element.tabulate_basis_gradients(
            reference_points_side
        )  # (F, 1 or Q, N_phi, 2)
        return jnp.broadcast_to(
            reference_gradients_side @ inverse_jacobian_side,
            (
                reference_points_side.shape[0],
                physical_points.shape[1],
                element.number_of_local_dofs,
                mesh.geometric_dimension,
            ),
        )

    gradients_plus = side_gradients(cells_plus)
    gradients_minus = side_gradients(cells_minus)

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
        gradients_plus=gradients_plus,
        gradients_minus=gradients_minus,
    )
