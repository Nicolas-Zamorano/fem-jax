"""
One particular function in a finite element space.
"""

from __future__ import annotations

import dataclasses

import jax

from jax_fem._shapes import check_shape
from jax_fem.space.finite_element_space import FiniteElementSpace


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FiniteElementFunction:
    """
    A function ``u_h = sum_i U_i phi_i`` in a finite element space.

    Attributes
    ----------
    space : FiniteElementSpace
        The space the function belongs to.
    dof_values : jax.Array
        Complete global DOF vector, including Dirichlet DOFs, shape
        ``(dofs_test, 1)``.
    """

    space: FiniteElementSpace
    dof_values: jax.Array

    def __post_init__(self) -> None:
        """
        Validate the shape of ``dof_values``.

        Raises
        ------
        ValueError
            If ``dof_values`` is not ``(space.number_of_dofs, 1)``.
        """
        check_shape(self.dof_values, (self.space.number_of_dofs, 1), "dof_values")
