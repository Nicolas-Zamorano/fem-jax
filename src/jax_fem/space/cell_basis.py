"""
Batched geometry and basis evaluation on one quadrature rule.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.reference_cell.quadrature import QuadratureRule
from jax_fem.space.finite_element_space import FiniteElementSpace


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class CellBasis:
    """
    Geometry and basis evaluation of a finite element space at one quadrature rule.

    Quantities that are mathematically constant along the cell axis or the
    quadrature axis (true for any affine, straight-sided triangle) are
    stored at their natural, un-repeated shape rather than broadcast out to
    the full ``(K, Q, ...)`` shape: they broadcast correctly wherever they
    are actually used (e.g. multiplied against the physical measure or
    against a genuinely ``(K, Q, ...)``-shaped coefficient).

    Attributes
    ----------
    space : FiniteElementSpace
        The finite element space being evaluated.
    quadrature : QuadratureRule
        The quadrature rule the space is evaluated at.
    physical_points : jax.Array
        Quadrature points mapped to physical coordinates, shape
        ``(K, Q, 1, d)``.
    jacobians : jax.Array
        Affine geometry map Jacobians, ``d(physical)/d(reference)``, shape
        ``(K, 1, d, d)``: constant in ``Q`` for straight-sided triangles.
    inverse_jacobians : jax.Array
        Shape ``(K, 1, d, d)``. Inverse of the affine geometry map Jacobians.
    jacobian_determinants : jax.Array
        Shape ``(K, 1, 1, 1)``. Determinant of the affine geometry map Jacobians.
    physical_weights : jax.Array
        Shape ``(K, Q, 1, 1)``. Physical weights for the quadrature rule.
    values : jax.Array
        Basis values at the quadrature points, shape ``(1, Q, N_phi, 1)``:
        constant in ``K`` for this scalar Lagrange element (reference basis
        values do not depend on which physical cell they are evaluated on).
    gradients : jax.Array
        Basis gradients with respect to physical coordinates, shape
        ``(K, 1, N_phi, d)``: constant in ``Q`` for straight-sided triangles.
    """

    space: FiniteElementSpace
    quadrature: QuadratureRule
    physical_points: jax.Array
    jacobians: jax.Array
    inverse_jacobians: jax.Array
    jacobian_determinants: jax.Array
    physical_weights: jax.Array
    values: jax.Array
    gradients: jax.Array

    def __post_init__(self) -> None:
        number_of_cells = self.space.cells_to_dofs.shape[0]
        number_of_points = self.quadrature.points.shape[1]
        number_of_local_dofs = self.space.element.number_of_local_dofs
        dimension = self.space.mesh.geometric_dimension

        check_shape(
            self.physical_points,
            (number_of_cells, number_of_points, 1, dimension),
            "physical_points",
        )
        check_shape(
            self.jacobians, (number_of_cells, 1, dimension, dimension), "jacobians"
        )
        check_shape(
            self.inverse_jacobians,
            (number_of_cells, 1, dimension, dimension),
            "inverse_jacobians",
        )
        check_shape(
            self.jacobian_determinants,
            (number_of_cells, 1, 1, 1),
            "jacobian_determinants",
        )
        check_shape(
            self.physical_weights,
            (number_of_cells, number_of_points, 1, 1),
            "physical_weights",
        )
        check_shape(
            self.values, (1, number_of_points, number_of_local_dofs, 1), "values"
        )
        check_shape(
            self.gradients,
            (number_of_cells, 1, number_of_local_dofs, dimension),
            "gradients",
        )


def create_cell_basis(
    space: FiniteElementSpace, quadrature: QuadratureRule
) -> CellBasis:
    """
    Evaluate a finite element space's geometry and basis at one quadrature rule.

    The affine geometry map is built from the mesh's corner vertices
    (``mesh.cells_to_vertices``), independent of the element's DOF map
    (``space.cells_to_dofs``). The two coincide for P1, but must not be
    conflated: the geometry mapping stays affine for straight-sided
    triangles even for future higher-order elements whose DOFs
    include edge or interior points that are not mesh vertices.

    Parameters
    ----------
    space: FiniteElementSpace
        The finite element space to evaluate.
    quadrature: QuadratureRule
        The quadrature rule to evaluate at, on a reference cell compatible
        with ``space.element.reference_cell``.

    Returns
    -------
    cell_basis : CellBasis
        The cell basis evaluated at the quadrature points.
    """
    mesh = space.mesh
    element = space.element
    reference_cell = quadrature.reference_cell

    # (K, 3, 1, 2): physical coordinates of each cell's 3 corner vertices.
    cell_vertex_coordinates = mesh.vertex_coordinates[mesh.cells_to_vertices]
    (
        cell_vertex_1_coordinates,
        cell_vertex_2_coordinates,
        cell_vertex_3_coodinates
    ) = jnp.split(cell_vertex_coordinates, 3, axis = -3)
    edge_1 = cell_vertex_2_coordinates - cell_vertex_1_coordinates
    edge_2 = cell_vertex_3_coodinates - cell_vertex_1_coordinates
    # Columns of the Jacobian are the physical images of the reference
    # triangle's two non-origin vertices: d(physical)/d(reference).
    # concatenate(axis=-2) stacks edge_1/edge_2 as rows; .mT makes them
    # columns instead, matching the row-vector convention used below
    # (physical = origin + reference @ J^T, grad_phys = grad_ref @ J^-1).
    jacobians = jnp.concatenate((edge_1, edge_2), axis=-2).mT  # (K, 1, 2, 2)
    inverse_jacobians = jnp.linalg.inv(jacobians)
    jacobian_determinants = jnp.expand_dims(jnp.linalg.det(jacobians), axis =(-1,-2))

    physical_weights = (quadrature.weights
        * reference_cell.measure
        * jacobian_determinants)

    # Row-vector points: x_row = origin_row + xhat_row @ J^T.
    physical_points = cell_vertex_1_coordinates + quadrature.points @ jacobians.mT

    reference_values = element.tabulate_basis_values(quadrature.points)
    reference_gradients = element.tabulate_basis_gradients(quadrature.points)

    values = reference_values
    # Physical gradients via the chain rule, row-vector convention:
    # grad_x = grad_xhat @ J^{-1}.
    gradients = reference_gradients @ inverse_jacobians

    return CellBasis(
        space=space,
        quadrature=quadrature,
        physical_points=physical_points,
        jacobians=jacobians,
        inverse_jacobians=inverse_jacobians,
        jacobian_determinants=jacobian_determinants,
        physical_weights=physical_weights,
        values=values,
        gradients=gradients,
    )
