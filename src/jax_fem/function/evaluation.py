"""
Evaluation of a finite element function on a ``CellBasis``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.space.cell_basis import CellBasis
from jax_fem.space.finite_element_space import gather_cell_dof_values


class FiniteElementFunctionEvaluation(NamedTuple):
    """
    A finite element function's values and gradients at quadrature points.

    Attributes
    ----------
    values : jax.Array
        Shape ``(K, Q, 1, 1)``.
    gradients : jax.Array
        Shape ``(K, 1, 1, d)``: constant in ``Q`` for straight-sided
        triangles (Section 6.3's collapsed-axis convention, ``CellBasis``);
        broadcasts against ``Q``-sized quantities wherever actually used.
    """

    values: jax.Array
    gradients: jax.Array


def evaluate_finite_element_function(
    function: FiniteElementFunction, basis: CellBasis
) -> FiniteElementFunctionEvaluation:
    """
    Evaluate a finite element function's values and gradients at ``basis``.

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
        The values and gradients of the finite element function at the
        quadrature points of the ``basis``.
    """
    if function.space is not basis.space:
        raise ValueError(
            "function and basis must share the same FiniteElementSpace "
            "instance."
        )

    local_dof_values = gather_cell_dof_values(function)
    local_dof_values = jnp.expand_dims(local_dof_values, axis = -3)

    values = basis.values.mT @ local_dof_values
    gradients = local_dof_values.mT @ basis.gradients

    return FiniteElementFunctionEvaluation(values=values, gradients=gradients)
