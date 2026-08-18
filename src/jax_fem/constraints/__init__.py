"""
Dirichlet condensation and full-vector reconstruction.
"""

from jax_fem.constraints.dirichlet import (
    CondensedSystem,
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)

__all__ = [
    "evaluate_dirichlet_dof_values",
    "CondensedSystem",
    "condense_dirichlet_system",
    "expand_condensed_solution",
]
