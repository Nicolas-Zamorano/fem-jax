"""Finite-element spaces and their quadrature-point evaluations (Section 10)."""

from jax_fem.space.cell_basis import CellBasis, create_cell_basis
from jax_fem.space.facet_basis import FacetBasis, create_facet_basis
from jax_fem.space.finite_element_space import (
    FiniteElementSpace,
    create_finite_element_space,
    find_boundary_dofs,
    gather_cell_dof_values,
)

__all__ = [
    "FiniteElementSpace",
    "create_finite_element_space",
    "find_boundary_dofs",
    "gather_cell_dof_values",
    "CellBasis",
    "create_cell_basis",
    "FacetBasis",
    "create_facet_basis",
]
