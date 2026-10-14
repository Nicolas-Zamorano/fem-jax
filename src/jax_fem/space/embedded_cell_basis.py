"""
Basis evaluation on planar cells embedded in a higher-dimensional space.
"""

from __future__ import annotations

import jax.numpy as jnp

from jax_fem.reference_cell.quadrature import QuadratureRule
from jax_fem.space.cell_basis import CellBasis
from jax_fem.space.finite_element_space import FiniteElementSpace


def create_embedded_cell_basis(
    space: FiniteElementSpace, quadrature: QuadratureRule
) -> CellBasis:
    """
    Evaluate a finite element space on planar cells embedded in ``R^g``.

    Parameters
    ----------
    space : FiniteElementSpace
        Space on an ``"identity"``-mapped element (P0, P1, P2), over a mesh
        such as ``DfnMesh``.
    quadrature : QuadratureRule
        Quadrature rule on a reference cell compatible with
        ``space.element.reference_cell``.

    Returns
    -------
    cell_basis : CellBasis
        The geometry and basis at the quadrature points. Its
        ``inverse_jacobians`` hold the pseudo-inverses ``J^+``, its
        ``jacobian_determinants`` the area scales ``sqrt(det(J^T J))``, and
        its ``gradients`` the tangential gradients.

    Raises
    ------
    ValueError
        If the element is not ``"identity"``-mapped.

    Notes
    -----
    The Jacobian ``J`` of a triangle in ``R^g`` is ``g x 2``, so its inverse
    and determinant generalize through the metric ``G = J^T J``:
    ``J^+ = G^-1 J^T`` and ``sqrt(det G)``. With the row-vector convention,
    physical points are ``origin + xhat @ J^T`` and gradients
    ``grad_ref @ J^+``, row vectors in ``R^g`` tangent to the cell. The area
    scale is unsigned, so the cell winding does not matter. For a square
    ``J`` this is ``create_cell_basis`` with ``|det J|`` in place of
    ``det J``.

    Contravariant Piola elements (RT0) are not supported: their fluxes
    across a trace shared by more than two cells need explicit coupling.
    """
    mesh = space.mesh
    element = space.element
    reference_cell = quadrature.reference_cell

    if element.mapping != "identity":
        raise ValueError(
            "create_embedded_cell_basis supports 'identity'-mapped elements "
            f"only, got element.mapping {element.mapping!r}."
        )

    cell_vertex_coordinates = mesh.vertex_coordinates[mesh.cells_to_vertices]
    origins, second_vertices, third_vertices = jnp.split(
        cell_vertex_coordinates, 3, axis=-3
    )
    jacobians = jnp.concatenate(
        (second_vertices - origins, third_vertices - origins), axis=-2
    ).mT
    metrics = jacobians.mT @ jacobians
    inverse_jacobians = jnp.linalg.solve(metrics, jacobians.mT)
    area_scales = jnp.expand_dims(jnp.sqrt(jnp.linalg.det(metrics)), axis=(-2, -1))

    physical_weights = quadrature.weights * reference_cell.measure * area_scales
    physical_points = origins + quadrature.points @ jacobians.mT
    reference_gradients = element.tabulate_basis_gradients(quadrature.points)

    return CellBasis(
        space=space,
        quadrature=quadrature,
        physical_points=physical_points,
        jacobians=jacobians,
        inverse_jacobians=inverse_jacobians,
        jacobian_determinants=area_scales,
        physical_weights=physical_weights,
        values=element.tabulate_basis_values(quadrature.points),
        gradients=reference_gradients @ inverse_jacobians,
        divergences=None,
    )
