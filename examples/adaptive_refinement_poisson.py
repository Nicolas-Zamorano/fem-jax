"""Adaptive-refinement (AMR) study, P1 Poisson on the L-shaped domain.

The AMR stretch item of Section 4.2 of the implementation plan
(``.context/implementation_plan.md``): a closed solve -> estimate -> mark ->
adapt loop on ``create_poisson_l_shaped_singular_problem`` (a genuine
re-entrant 270-degree corner singularity), driven by the classical residual
error estimator (``compute_residual_error_estimator``) and Dorfler bulk
marking (``mark_cells_by_dorfler_bulk_criterion``).

At every iteration, three figures are saved: the 2D and 3D error plots (as
in ``examples/uniform_refinement_poisson.py``), plus a new one specific to
AMR -- a 2D mesh plot of the estimator ``eta_K`` and the cells marked for
refinement (``plot_error_estimator_and_marked_cells``). At the end, the same
kind of convergence table is written, using the mean cell diameter as an
effective ``h`` (a graded AMR mesh has no single mesh size).

Compare its convergence table against
``examples/uniform_refinement_poisson.py``'s: AMR concentrates DOFs at the
corner instead of refining the whole domain, which is the entire point of
adaptivity for a singular solution.

Run with:

    uv run python examples/adaptive_refinement_poisson.py
"""

import csv
import math
from pathlib import Path

import jax.numpy as jnp

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import (
    compute_h1_seminorm_error,
    compute_l2_error,
    compute_residual_error_estimator,
    plot_error_estimator_and_marked_cells,
    plot_fem_error,
    plot_fem_error_3d,
    save_plots,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import (
    adapt_l_shaped_mesh_from_error_estimator,
    cell_diameters,
    create_gmsh_l_shaped_mesh,
    mark_cells_by_dorfler_bulk_criterion,
)
from jax_fem.problem import (
    create_poisson_l_shaped_singular_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from jax_fem.space.facet_basis import create_facet_basis

_TABLE_COLUMNS = [
    "iteration",
    "n_dofs",
    "mean_h",
    "l2_error",
    "h1_error",
    "eta",
    "l2_ratio",
    "h1_ratio",
    "indicator_fraction",
]


def _solve_and_estimate(mesh, element):
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
    problem = create_poisson_l_shaped_singular_problem(mesh)

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full_dof_values = expand_condensed_solution(free_dof_values, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full_dof_values)

    facet_quadrature = ReferenceInterval().create_quadrature(4)
    facet_basis = create_facet_basis(basis, facet_quadrature)
    cell_indicators, eta = compute_residual_error_estimator(
        solution, basis, facet_basis, problem
    )

    l2_error = float(compute_l2_error(solution, basis, problem))
    h1_error = float(compute_h1_seminorm_error(solution, basis, problem))
    mean_h = float(jnp.mean(cell_diameters(mesh)))

    return space, basis, problem, solution, cell_indicators, float(eta), l2_error, h1_error, mean_h


def _print_and_save_table(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'iter':>5} {'N_dofs':>8} {'mean h':>10} {'L2 error':>12} {'H1 error':>12} "
        f"{'eta':>12} {'L2 ratio':>10} {'H1 ratio':>10} {'indicator fraction':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        l2_ratio = row["l2_ratio"]
        h1_ratio = row["h1_ratio"]
        l2_ratio_str = "-" if math.isnan(l2_ratio) else f"{l2_ratio:.3f}"
        h1_ratio_str = "-" if math.isnan(h1_ratio) else f"{h1_ratio:.3f}"
        print(
            f"{row['iteration']:>5} {row['n_dofs']:>8} {row['mean_h']:>10.5f} "
            f"{row['l2_error']:>12.6e} {row['h1_error']:>12.6e} {row['eta']:>12.6e} "
            f"{l2_ratio_str:>10} {h1_ratio_str:>10} {row['indicator_fraction']:>12.6e}"
        )


def main(
    base_h: float = 0.5,
    iterations: int = 10,
    theta: float = 0.6,
    refinement_factor: float = 0.5,
) -> None:
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    mesh = create_gmsh_l_shaped_mesh(base_h)

    plots: dict = {}
    rows: list[dict] = []
    for iteration in range(iterations):
        (
            space,
            basis,
            problem,
            solution,
            cell_indicators,
            eta,
            l2_error,
            h1_error,
            mean_h,
        ) = _solve_and_estimate(mesh, element)
        marked_cells = mark_cells_by_dorfler_bulk_criterion(cell_indicators, theta=theta)

        plots[f"iter_{iteration}_error_mesh"] = plot_fem_error(
            solution, basis, problem, mesh
        )
        plots[f"iter_{iteration}_error_mesh_3d"] = plot_fem_error_3d(
            solution, basis, problem, mesh
        )
        plots[f"iter_{iteration}_estimator_and_marked"] = (
            plot_error_estimator_and_marked_cells(cell_indicators, marked_cells, mesh)
        )

        rows.append(
            {
                "iteration": iteration,
                "n_dofs": space.number_of_dofs,
                "mean_h": mean_h,
                "l2_error": l2_error,
                "h1_error": h1_error,
                "eta": eta,
                "l2_ratio": float("nan"),
                "h1_ratio": float("nan"),
                "indicator_fraction": float(jnp.sum(cell_indicators[marked_cells]**2)/jnp.sum(cell_indicators**2))
            }
        )
        print(
            f"iteration {iteration}: N_dofs={space.number_of_dofs:5d}  "
            f"mean h={mean_h:.5f}  L2={l2_error:.6e}  H1={h1_error:.6e}  "
            f"eta={eta:.6e}  marked={int(jnp.sum(marked_cells))}/"
            f"{mesh.cells_to_vertices.shape[0]}"
        )

        if iteration < iterations - 1:
            mesh = adapt_l_shaped_mesh_from_error_estimator(
                mesh,
                cell_indicators,
                theta=theta,
                refinement_factor=refinement_factor,
            )

    for previous, current in zip(rows, rows[1:], strict=False):
        current["l2_ratio"] = previous["l2_error"] / current["l2_error"]
        current["h1_ratio"] = previous["h1_error"] / current["h1_error"]

    directory = save_plots(
        plots, directory="experiments", name="adaptive_refinement_poisson_l_shaped"
    )
    _print_and_save_table(rows, directory / "convergence_table.csv")
    print("\nPlots and table saved to", directory)


if __name__ == "__main__":
    main()
