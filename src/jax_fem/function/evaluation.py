"""
Evaluation of a finite element function on a ``CellBasis``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.space.cell_basis import CellBasis
from jax_fem.space.finite_element_space import gather_cell_dof_values


class FiniteElementFunctionEvaluation(NamedTuple):
    """
    A finite element function's values and gradient/divergence at quadrature
    points.

    Exactly one of ``gradients``/``divergence`` is non-``None``, matching
    ``CellBasis``'s own ``gradients``/``divergences`` duality
    (``space.element.mapping``, Section 5.2): ``gradients`` for an
    ``"identity"``-mapped (scalar) element, ``divergence`` for a
    ``"contravariant_piola"``-mapped (vector) element.

    Attributes
    ----------
    values : jax.Array
        Shape ``(K, Q, 1, value_dim)`` (``value_dim = 1`` for a scalar
        element, e.g. P1/P2/P0; ``value_dim = 2`` for a vector element, e.g.
        RT0).
    gradients : jax.Array | None
        Shape ``(K, 1, 1, d)``: constant in ``Q`` for straight-sided
        triangles (Section 6.3's collapsed-axis convention, ``CellBasis``);
        broadcasts against ``Q``-sized quantities wherever actually used.
        ``None`` for a vector element.
    divergence : jax.Array | None
        Shape ``(K, 1, 1, 1)`` or ``(K, Q, 1, 1)`` (same constant-vs-varying
        duality). ``None`` for a scalar element.
    """

    values: jax.Array
    gradients: jax.Array | None
    divergence: jax.Array | None


def evaluate_finite_element_function(
    function: FiniteElementFunction, basis: CellBasis
) -> FiniteElementFunctionEvaluation:
    """
    Evaluate a finite element function's values and gradient/divergence at
    ``basis``.

    Parameters
    ----------
    function : FiniteElementFunction
        The finite element function to evaluate. Must belong to the same
        ``FiniteElementSpace`` as ``basis``.
    basis : CellBasis
        The quadrature-point geometry and basis to evaluate at.

    Returns
    -------
    function_evaluation : FiniteElementFunctionEvaluation
        The values and gradient/divergence of the finite element function
        at the quadrature points of the ``basis``.
    """
    if function.space is not basis.space:
        raise ValueError(
            "function and basis must share the same FiniteElementSpace "
            "instance."
        )

    local_dof_values = gather_cell_dof_values(function)
    local_dof_values = local_dof_values[:, None, :, :]  # (K, 1, N_phi, 1)

    # local_dof_values.mT @ basis.values (rather than basis.values.mT @
    # local_dof_values): for a scalar element these are numerically
    # identical (both compute the same dot product over N_phi), but only
    # this order also generalizes correctly to a vector element's
    # (K, Q, N_phi, value_dim) values -- basis.values.mT @ local_dof_values
    # would instead produce a (value_dim, 1) *column*, the wrong shape for
    # the row-vector convention used throughout (Section 6.2).
    values = local_dof_values.mT @ basis.values  # (K, Q, 1, value_dim)

    gradients = (
        local_dof_values.mT @ basis.gradients if basis.gradients is not None else None
    )
    divergence = (
        local_dof_values.mT @ basis.divergences
        if basis.divergences is not None
        else None
    )

    return FiniteElementFunctionEvaluation(
        values=values, gradients=gradients, divergence=divergence
    )
