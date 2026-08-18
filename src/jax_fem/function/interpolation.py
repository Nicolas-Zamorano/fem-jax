"""
Nodal interpolation into a finite element space.
"""

from __future__ import annotations

from jax_fem._shapes import check_shape
from jax_fem.element.lagrange_triangle import LagrangeTriangleP1
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.problem.elliptic import ScalarCoefficient
from jax_fem.space.finite_element_space import FiniteElementSpace


def interpolate_function(
    function: ScalarCoefficient, space: FiniteElementSpace
) -> FiniteElementFunction:
    """
    Interpolate a scalar physical function into a finite element space.
    
    Parameters
    ----------
    function : ScalarCoefficient
        The physical function to interpolate, receives ``(*batch, 1, d)`` and
        returns ``(*batch, 1, 1)``.
    space : FiniteElementSpace
        The finite element space to interpolate into.

    Returns
    -------
    FiniteElementFunction
        ``dof_values`` has shape ``(N_dofs, 1)``.
    """
    if not isinstance(space.element, LagrangeTriangleP1):
        raise NotImplementedError(
            "interpolate_function currently only supports nodal "
            "interpolation for LagrangeTriangleP1. Future non-nodal "
            "elements will interpolate via their own interpolation "
            "functionals (Section 11.3)."
        )

    values = function(space.dof_coordinates)
    check_shape(values, (space.number_of_dofs, 1, 1), "function(...)")
    dof_values = values[:, 0, :]

    return FiniteElementFunction(space=space, dof_values=dof_values)
