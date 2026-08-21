"""
Nodal interpolation into a finite element space.
"""

from __future__ import annotations

from jax_fem._shapes import check_shape
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.problem.elliptic import ScalarCoefficient
from jax_fem.space.finite_element_space import FiniteElementSpace


def interpolate_function(
    function: ScalarCoefficient, space: FiniteElementSpace
) -> FiniteElementFunction:
    """
    Interpolate a scalar physical function into a finite element space.

    Generic over any nodal element (``space.element.is_nodal``, e.g.
    ``LagrangeTriangleP1`` or ``LagrangeTriangleP2``): every DOF is a point
    evaluation at its ``space.dof_coordinates`` entry, so interpolation is
    just evaluating ``function`` there directly, independent of element
    degree.

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
    if not space.element.is_nodal:
        raise NotImplementedError(
            "interpolate_function only supports nodal elements "
            "(element.is_nodal == True). A non-nodal element (e.g. a "
            "future H(div) element with edge-flux-moment DOFs) requires "
            "its own interpolation functionals (Section 11.3)."
        )

    values = function(space.dof_coordinates)
    check_shape(values, (space.number_of_dofs, 1, 1), "function(...)")
    dof_values = values[:, 0, :]

    return FiniteElementFunction(space=space, dof_values=dof_values)
