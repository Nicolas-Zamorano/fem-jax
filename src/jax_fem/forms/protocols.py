"""
Form protocols.

Forms are independent of a particular mesh and element: they operate on
already evaluated basis functions, fields, and problem coefficients. No
symbolic expression language is required; forms are pure Python callables
satisfying these small protocols.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax

if TYPE_CHECKING:
    from jax_fem.function.evaluation import FiniteElementFunctionEvaluation
    from jax_fem.problem.elliptic import EllipticProblem
    from jax_fem.space.cell_basis import CellBasis


@runtime_checkable
class BilinearForm(Protocol):
    """
    A pointwise bilinear form.

    Notes
    -----
    The test and trial bases may belong to different spaces, but must share
    the same cells and quadrature points. ``coefficients`` is any pytree the
    form needs (e.g. ``EllipticCoefficientValues``), or ``None``.
    """

    def __call__(
        self,
        test_basis: CellBasis,
        trial_basis: CellBasis,
        coefficients: Any,
    ) -> jax.Array:
        """
        Evaluate the form at every cell and quadrature point.

        Parameters
        ----------
        test_basis : CellBasis
            Evaluated test basis.
        trial_basis : CellBasis
            Evaluated trial basis.
        coefficients : Any
            Data the form needs, evaluated at the quadrature points.

        Returns
        -------
        pointwise_values : jax.Array
            Pointwise form values, shape ``(K, Q, N_test, N_trial)``.
        """
        ...


@runtime_checkable
class LinearForm(Protocol):
    """
    A pointwise linear form.

    Notes
    -----
    ``coefficients`` is any pytree the form needs (e.g.
    ``EllipticCoefficientValues``), or ``None``.
    """

    def __call__(self, test_basis: CellBasis, coefficients: Any) -> jax.Array:
        """
        Evaluate the form at every cell and quadrature point.

        Parameters
        ----------
        test_basis : CellBasis
            Evaluated test basis.
        coefficients : Any
            Data the form needs, evaluated at the quadrature points.

        Returns
        -------
        pointwise_values : jax.Array
            Pointwise form values, shape ``(K, Q, N_test, 1)``.
        """
        ...


@runtime_checkable
class Functional(Protocol):
    """A pointwise scalar functional."""

    def __call__(
        self,
        basis: CellBasis,
        fields: Mapping[str, FiniteElementFunctionEvaluation],
        problem: EllipticProblem,
    ) -> jax.Array:
        """
        Evaluate the functional at every cell and quadrature point.

        Parameters
        ----------
        basis : CellBasis
            Evaluated basis.
        fields : Mapping[str, FiniteElementFunctionEvaluation]
            Finite element functions evaluated on ``basis``.
        problem : EllipticProblem
            The elliptic problem.

        Returns
        -------
        pointwise_values : jax.Array
            Pointwise functional values, shape ``(K, Q, 1, 1)``.
        """
        ...
