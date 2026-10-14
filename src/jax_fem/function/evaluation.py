"""
Evaluation of a finite element function on a ``CellBasis``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

from jax_fem.assembly.global_assembly import gather_vector
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.space.cell_basis import CellBasis


class FiniteElementFunctionEvaluation(NamedTuple):
    """
    Values and derivatives of a finite element function at quadrature points.

    Attributes
    ----------
    values : jax.Array
        Function values, shape ``(K, Q, 1, value_dim)``; ``value_dim`` is 1
        for a scalar element (P0, P1, P2) and 2 for a vector element (RT0).
    gradients : jax.Array | None
        Physical gradients, shape ``(K, 1, 1, d)`` when the element's
        gradients are constant on each cell (P0, P1) or ``(K, Q, 1, d)``
        otherwise (P2). ``None`` for a vector element.
    divergence : jax.Array | None
        Physical divergence, shape ``(K, 1, 1, 1)`` or ``(K, Q, 1, 1)``.
        ``None`` for a scalar element.

    Notes
    -----
    Exactly one of ``gradients`` and ``divergence`` is set, following the
    element mapping: ``gradients`` for ``"identity"``-mapped elements and
    ``divergence`` for ``"contravariant_piola"``-mapped elements.
    """

    values: jax.Array
    gradients: jax.Array | None
    divergence: jax.Array | None


def evaluate_finite_element_function(
    function: FiniteElementFunction, basis: CellBasis
) -> FiniteElementFunctionEvaluation:
    """
    Evaluate a finite element function at the quadrature points of a basis.

    Parameters
    ----------
    function : FiniteElementFunction
        The finite element function to evaluate. Must belong to the same
        ``FiniteElementSpace`` instance as ``basis``.
    basis : CellBasis
        The evaluated basis to evaluate at.

    Returns
    -------
    function_evaluation : FiniteElementFunctionEvaluation
        Values and gradients or divergence at the quadrature points.

    Notes
    -----
    Local DOF values multiply the basis from the left
    (``local_dof_values.mT @ basis.values``). For a scalar element this is
    the same as the transposed product, but only this order gives the row
    vector ``(K, Q, 1, value_dim)`` for a vector-valued element.
    """
    if function.space is not basis.space:
        raise ValueError(
            "function and basis must share the same FiniteElementSpace "
            "instance."
        )

    local_dof_values = gather_vector(function.dof_values, function.space).mT
    values = local_dof_values @ basis.values
    gradients = (
        local_dof_values @ basis.gradients if basis.gradients is not None else None
    )
    divergence = (
        local_dof_values @ basis.divergences
        if basis.divergences is not None
        else None
    )

    return FiniteElementFunctionEvaluation(
        values=values, gradients=gradients, divergence=divergence
    )
