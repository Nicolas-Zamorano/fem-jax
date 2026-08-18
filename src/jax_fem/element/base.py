"""
FiniteElement interface shared by all finite elements.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax

from jax_fem.reference_cell.base import ReferenceCell


@runtime_checkable
class FiniteElement(Protocol):
    """
    Local approximation space, local basis, and local DOFs on a reference cell.

    A ``FiniteElement`` describes the reference-cell finite element only. It
    owns local basis functions and local degrees of freedom; it owns
    neither global numbering nor assembly. It must also
    describe which reference-cell entities own its local DOFs (documented
    per concrete element, since the entity-DOF description differs by
    element family); that information is consumed by
    ``FiniteElementSpace``.

    Attributes
    ----------
    reference_cell : ReferenceCell
        The reference cell this element is defined on.
    polynomial_degree: int
        Polynomial degree of the local approximation space.
    number_of_local_dofs: int
        Number of local degrees of freedom, ``N_phi``.
    value_shape : tuple[int, ...]
        Shape of one basis function's value; ``(1,)`` for a scalar element.
    """

    reference_cell: ReferenceCell
    polynomial_degree: int
    number_of_local_dofs: int
    value_shape: tuple[int, ...]

    def tabulate_basis_values(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate all local basis functions at reference points.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(Q, 1, d)``.

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(Q, N_phi, 1)``.
        """
        ...

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate the reference gradient of all local basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(Q, 1, d)``.

        Returns
        -------
        basis_gradients : jax.Array
            Basis gradients with respect to reference coordinates, shape
            ``(Q, N_phi, d)``.
        """
        ...
