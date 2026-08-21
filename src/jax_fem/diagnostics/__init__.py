"""
Exact-error functionals, norms, and a posteriori error estimation for
finite element solutions.
"""

from jax_fem.diagnostics.errors import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_l2_error,
    compute_relative_l2_error,
)
from jax_fem.diagnostics.estimator import compute_residual_error_estimator
from jax_fem.diagnostics.norms import (
    energy_error_density,
    h1_seminorm_error_density,
    l2_error_density,
)
from jax_fem.diagnostics.plotting import plot_fem_error, plot_fem_error_3d, save_plots

__all__ = [
    "compute_l2_error",
    "compute_relative_l2_error",
    "compute_h1_seminorm_error",
    "compute_energy_error",
    "compute_residual_error_estimator",
    "l2_error_density",
    "h1_seminorm_error_density",
    "energy_error_density",
    "plot_fem_error",
    "plot_fem_error_3d",
    "save_plots",
]
