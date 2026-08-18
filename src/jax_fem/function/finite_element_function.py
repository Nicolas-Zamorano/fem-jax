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
    """One particular function ``u_h = sum_i U_i Phi_i`` in a ``FiniteElementSpace``.

    ``dof_values`` is always the complete vector, including constrained
    Dirichlet DOFs.

    Attributes
    ----------
    space : FiniteElementSpace
        The finite element space this function belongs to.
    dof_values : jax.Array
        The complete global DOF vector, shape ``(N_dofs, 1)``.
    """

    space: FiniteElementSpace
    dof_values: jax.Array

    def __post_init__(self) -> None:
        check_shape(self.dof_values, (self.space.number_of_dofs, 1), "dof_values")
