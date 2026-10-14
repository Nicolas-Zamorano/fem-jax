"""
Nodal interpolation into a finite element space.
"""

from __future__ import annotations

import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.problem.elliptic import ScalarCoefficient
from jax_fem.space.finite_element_space import FiniteElementSpace


def interpolate_function(
    function: ScalarCoefficient, space: FiniteElementSpace
) -> FiniteElementFunction:
    """
    Nodal interpolant of a scalar function.

    Parameters
    ----------
    function : ScalarCoefficient
        Function to interpolate; takes points of shape ``(*batch, 1, d)`` and
        returns ``(*batch, 1, 1)``.
    space : FiniteElementSpace
        Space on a nodal element.

    Returns
    -------
    interpolant : FiniteElementFunction
        The interpolant, with ``dof_values`` of shape ``(dofs_test, 1)``.

    Notes
    -----
    Every DOF of a nodal element is point evaluation at its
    ``space.dof_coordinates`` entry, so interpolation evaluates ``function``
    there, for any element degree.
    """
    if not space.element.is_nodal:
        raise NotImplementedError(
            "interpolate_function only supports nodal elements "
            "(element.is_nodal == True). A non-nodal element, such as an "
            "H(div) element with edge-flux DOFs, needs its own interpolation "
            "functionals."
        )

    values = function(space.dof_coordinates)
    check_shape(values, (space.number_of_dofs, 1, 1), "function(...)")
    dof_values = jnp.squeeze(values, axis=-2)

    return FiniteElementFunction(space=space, dof_values=dof_values)
