"""
Form protocols.

Forms are independent of a particular mesh and element: they operate on
already evaluated basis functions, fields, and problem coefficients. No
symbolic expression language is required; forms are pure Python callables
satisfying these small protocols.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax

if TYPE_CHECKING:
    from jax_fem.function.evaluation import FiniteElementFunctionEvaluation
    from jax_fem.problem.elliptic import EllipticCoefficientValues, EllipticProblem
    from jax_fem.space.cell_basis import CellBasis


@runtime_checkable
class BilinearForm(Protocol):
    """A pointwise bilinear form integrand."""

    def __call__(
        self,
        basis: CellBasis,
        coefficients: EllipticCoefficientValues,
    ) -> jax.Array:
        """Return shape ``(K, Q, N_phi, N_phi)``."""
        ...


@runtime_checkable
class LinearForm(Protocol):
    """A pointwise linear form integrand."""

    def __call__(
        self,
        basis: CellBasis,
        coefficients: EllipticCoefficientValues,
    ) -> jax.Array:
        """Return shape ``(K, Q, N_phi, 1)``."""
        ...


@runtime_checkable
class Functional(Protocol):
    """A pointwise scalar functional integrand."""

    def __call__(
        self,
        basis: CellBasis,
        fields: Mapping[str, FiniteElementFunctionEvaluation],
        problem: EllipticProblem,
    ) -> jax.Array:
        """Return a pointwise scalar array of shape ``(K, Q, 1, 1)``."""
        ...
