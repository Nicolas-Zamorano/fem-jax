"""
Error functionals and norms, a posteriori error estimation, and plotting.
"""

from jax_fem.diagnostics.errors import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_l2_error,
)
from jax_fem.diagnostics.estimator import (
    compute_element_residual_indicators,
    compute_facet_jump_indicators,
    compute_residual_error_estimator,
    evaluate_facet_fluxes,
    evaluate_flux_divergence,
)
from jax_fem.diagnostics.mixed_errors import (
    compute_flux_divergence_l2_error,
    compute_flux_l2_error,
)
from jax_fem.diagnostics.mixed_norms import (
    flux_divergence_l2_error_functional,
    flux_l2_error_functional,
)
from jax_fem.diagnostics.norms import (
    energy_error_functional,
    h1_seminorm_error_functional,
    l2_error_functional,
)
from jax_fem.diagnostics.plotting import (
    create_timestamped_output_directory,
    plot_convergence_rates,
    plot_error_estimator_and_marked_cells,
    plot_fem_error,
    plot_fem_error_3d,
    plot_fracture_solutions,
    plot_mixed_solution,
    plot_trace_fluxes,
    save_plot,
    save_plots,
)
from jax_fem.diagnostics.trace_fluxes import TraceFluxes, evaluate_trace_fluxes

__all__ = [
    "compute_l2_error",
    "compute_h1_seminorm_error",
    "compute_energy_error",
    "compute_residual_error_estimator",
    "compute_element_residual_indicators",
    "compute_facet_jump_indicators",
    "evaluate_flux_divergence",
    "evaluate_facet_fluxes",
    "compute_flux_l2_error",
    "compute_flux_divergence_l2_error",
    "l2_error_functional",
    "h1_seminorm_error_functional",
    "energy_error_functional",
    "flux_l2_error_functional",
    "flux_divergence_l2_error_functional",
    "TraceFluxes",
    "evaluate_trace_fluxes",
    "plot_fem_error",
    "plot_fem_error_3d",
    "plot_error_estimator_and_marked_cells",
    "plot_mixed_solution",
    "plot_fracture_solutions",
    "plot_trace_fluxes",
    "plot_convergence_rates",
    "save_plot",
    "save_plots",
    "create_timestamped_output_directory",
]
