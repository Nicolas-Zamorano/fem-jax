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
from jax_fem.diagnostics.mixed_errors import (
    compute_flux_divergence_l2_error,
    compute_flux_l2_error,
)
from jax_fem.diagnostics.mixed_norms import (
    flux_divergence_l2_error_density,
    flux_l2_error_density,
)
from jax_fem.diagnostics.norms import (
    energy_error_density,
    h1_seminorm_error_density,
    l2_error_density,
)
from jax_fem.diagnostics.plotting import (
    plot_error_estimator_and_marked_cells,
    plot_fem_error,
    plot_fem_error_3d,
    plot_mixed_solution,
    save_plots,
)

__all__ = [
    "compute_l2_error",
    "compute_relative_l2_error",
    "compute_h1_seminorm_error",
    "compute_energy_error",
    "compute_residual_error_estimator",
    "compute_flux_l2_error",
    "compute_flux_divergence_l2_error",
    "l2_error_density",
    "h1_seminorm_error_density",
    "energy_error_density",
    "flux_l2_error_density",
    "flux_divergence_l2_error_density",
    "plot_fem_error",
    "plot_fem_error_3d",
    "plot_error_estimator_and_marked_cells",
    "plot_mixed_solution",
    "save_plots",
]
