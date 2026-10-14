"""Finite element spaces and their evaluation at quadrature points."""

from jax_fem.space.cell_basis import (
    CellBasis,
    compute_tangent_projectors,
    create_cell_basis,
)
from jax_fem.space.embedded_cell_basis import create_embedded_cell_basis
from jax_fem.space.facet_basis import FacetBasis, create_facet_basis
from jax_fem.space.finite_element_space import (
    FiniteElementSpace,
    create_finite_element_space,
    find_boundary_dofs,
)

__all__ = [
    "FiniteElementSpace",
    "create_finite_element_space",
    "find_boundary_dofs",
    "CellBasis",
    "create_cell_basis",
    "create_embedded_cell_basis",
    "compute_tangent_projectors",
    "FacetBasis",
    "create_facet_basis",
]
