"""
Finite element functions: representation, evaluation, interpolation.
"""

from jax_fem.function.evaluation import (
    FiniteElementFunctionEvaluation,
    evaluate_finite_element_function,
)
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.function.interpolation import interpolate_function

__all__ = [
    "FiniteElementFunction",
    "FiniteElementFunctionEvaluation",
    "evaluate_finite_element_function",
    "interpolate_function",
]
