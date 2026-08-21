"""Adaptive-refinement (AMR) study, P1 Poisson, any manufactured problem.

The AMR stretch item of Section 4.2 of the implementation plan
(``.context/implementation_plan.md``) and
``context/in_house_amr_implementation_plan.md``: a closed solve -> estimate
-> mark -> adapt loop on any of ``jax_fem.problem``'s
manufactured elliptic problems (except the mixed/dual formulation ones --
those have no residual estimator here), driven by the classical residual
error estimator (``compute_residual_error_estimator``), Dorfler bulk
marking (``mark_cells_by_dorfler_bulk_criterion``), and the in-house LEB
engine (``adapt_mesh_from_error_estimator``). Every problem lives on the
unit square except ``l_shaped_singular`` (a genuine re-entrant 270-degree
corner singularity), which lives on the L-shaped domain -- the right base
mesh factory is picked automatically from ``problem_name``.

At every iteration, three figures are saved (and closed) immediately
rather than accumulating in memory across the whole run: the 2D and 3D
error plots (as in ``examples/uniform_refinement_poisson.py``), plus a new
one specific to AMR -- a 2D mesh plot of the estimator ``eta_K`` and the
cells marked for refinement (``plot_error_estimator_and_marked_cells``).
At the end, the same kind of convergence table is written, using the mean
cell diameter as an effective ``h`` (a graded AMR mesh has no single mesh
size).

Compare its convergence table against
``examples/uniform_refinement_poisson.py``'s: AMR concentrates DOFs at the
singularity instead of refining the whole domain, which is the entire
point of adaptivity for a singular solution.

Run with:

    uv run python examples/adaptive_refinement_poisson.py
"""

import csv
import math
from collections.abc import Callable
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
    create_timestamped_output_directory,
    plot_convergence_rates,
    plot_error_estimator_and_marked_cells,
    plot_fem_error,
    plot_fem_error_3d,
    save_plot,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import (
    TriangleMesh,
    adapt_mesh_from_error_estimator,
    cell_diameters,
    create_gmsh_l_shaped_mesh,
    create_gmsh_unit_square_mesh,
    mark_cells_by_dorfler_bulk_criterion,
)
from jax_fem.problem import (
    EllipticProblem,
    create_poisson_cos_sin_problem,
    create_poisson_exponential_problem,
    create_poisson_l_shaped_singular_problem,
    create_poisson_sin_sin_problem,
    create_poisson_singular_problem,
    create_simple_elliptic_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from jax_fem.space.facet_basis import create_facet_basis

# Every problem below lives on the unit square except "l_shaped_singular",
# which needs the genuine re-entrant-corner L-shaped domain -- see
# ``_base_mesh_for_problem``.
_UNIT_SQUARE_PROBLEM_FACTORIES: dict[str, Callable[[TriangleMesh], EllipticProblem]] = {
    "sin_sin": create_poisson_sin_sin_problem,
    "cos_sin": create_poisson_cos_sin_problem,
    "exponential": create_poisson_exponential_problem,
    "singular": create_poisson_singular_problem,
    "simple_elliptic": create_simple_elliptic_problem,
}
_L_SHAPED_PROBLEM_FACTORIES: dict[str, Callable[[TriangleMesh], EllipticProblem]] = {
    "l_shaped_singular": create_poisson_l_shaped_singular_problem,
}
_PROBLEM_FACTORIES: dict[str, Callable[[TriangleMesh], EllipticProblem]] = {
    **_UNIT_SQUARE_PROBLEM_FACTORIES,
    **_L_SHAPED_PROBLEM_FACTORIES,
}

_TABLE_COLUMNS = [
    "iteration",
    "n_dofs",
    "mean_h",
    "l2_error",
    "h1_error",
    "eta",
    "l2_ratio",
    "h1_ratio",
    "l2_rate",
    "h1_rate",
    "indicator_fraction",
]


def _problem_factory(problem_name: str) -> Callable[[TriangleMesh], EllipticProblem]:
    try:
        return _PROBLEM_FACTORIES[problem_name]
    except KeyError:
        raise ValueError(
            f"Unknown problem {problem_name!r}; choose one of "
            f"{sorted(_PROBLEM_FACTORIES)}."
        ) from None


def _base_mesh_for_problem(problem_name: str, base_h: float) -> TriangleMesh:
    """The right base-domain mesh for ``problem_name``.

    ``l_shaped_singular`` needs the L-shaped domain; every other problem
    lives on the unit square. ``base_h`` is a target element size either
    way -- the unit-square factory instead wants an integer cell count per
    side, so it's converted via ``round(1 / base_h)``.
    """
    if problem_name in _L_SHAPED_PROBLEM_FACTORIES:
        return create_gmsh_l_shaped_mesh(base_h)
    number_of_cells_per_side = max(1, round(1.0 / base_h))
    return create_gmsh_unit_square_mesh(number_of_cells_per_side)


def _solve_and_estimate(mesh, element, problem_factory):
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
    problem = problem_factory(mesh)

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

    return (
        space,
        basis,
        problem,
        solution,
        cell_indicators,
        float(eta),
        l2_error,
        h1_error,
        mean_h,
    )


def _fmt(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.3f}"


def _print_and_save_table(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'iter':>5} {'N_dofs':>8} {'mean h':>10} {'L2 error':>12} "
        f"{'H1 error':>12} {'eta':>12} {'L2 ratio':>10} {'H1 ratio':>10} "
        f"{'L2 rate':>10} {'H1 rate':>10} {'indicator fraction':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['iteration']:>5} {row['n_dofs']:>8} {row['mean_h']:>10.5f} "
            f"{row['l2_error']:>12.6e} {row['h1_error']:>12.6e} {row['eta']:>12.6e} "
            f"{_fmt(row['l2_ratio']):>10} {_fmt(row['h1_ratio']):>10} "
            f"{_fmt(row['l2_rate']):>10} {_fmt(row['h1_rate']):>10} "
            f"{row['indicator_fraction']:>12.6e}"
        )


def main(
    problem_name: str = "l_shaped_singular",
    base_h: float = 0.5,
    iterations: int = 20,
    theta: float = 0.6,
) -> None:
    problem_factory = _problem_factory(problem_name)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    mesh = _base_mesh_for_problem(problem_name, base_h)

    directory = create_timestamped_output_directory(
        "experiments", f"adaptive_refinement_poisson_{problem_name}"
    )
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
        ) = _solve_and_estimate(mesh, element, problem_factory)
        marked_cells = mark_cells_by_dorfler_bulk_criterion(
            cell_indicators, theta=theta
        )

        save_plot(
            plot_fem_error(solution, basis, problem, mesh),
            directory / f"iter_{iteration}_error_mesh.png",
        )
        save_plot(
            plot_fem_error_3d(solution, basis, problem, mesh),
            directory / f"iter_{iteration}_error_mesh_3d.png",
        )
        save_plot(
            plot_error_estimator_and_marked_cells(cell_indicators, marked_cells, mesh),
            directory / f"iter_{iteration}_estimator_and_marked.png",
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
                "l2_rate": float("nan"),
                "h1_rate": float("nan"),
                "indicator_fraction": float(
                    jnp.sum(cell_indicators[marked_cells] ** 2)
                    / jnp.sum(cell_indicators**2)
                ),
            }
        )
        print(
            f"iteration {iteration}: N_dofs={space.number_of_dofs:5d}  "
            f"mean h={mean_h:.5f}  L2={l2_error:.6e}  H1={h1_error:.6e}  "
            f"eta={eta:.6e}  marked={int(jnp.sum(marked_cells))}/"
            f"{mesh.cells_to_vertices.shape[0]}"
        )

        if iteration < iterations - 1:
            mesh = adapt_mesh_from_error_estimator(mesh, cell_indicators, theta=theta)

    for previous, current in zip(rows, rows[1:], strict=False):
        current["l2_ratio"] = previous["l2_error"] / current["l2_error"]
        current["h1_ratio"] = previous["h1_error"] / current["h1_error"]
        # The observed local convergence rate between consecutive iterations:
        # the exponent p such that error ~ n_dofs^p, i.e. the number
        # directly comparable against theory's target rate (e.g. -1/2 for
        # H1 under optimally-graded AMR), unlike the raw ratio above, which
        # conflates the error's decay with however much n_dofs happened to
        # grow that particular iteration.
        log_n_dofs_ratio = math.log(current["n_dofs"] / previous["n_dofs"])
        if log_n_dofs_ratio > 0.0:
            current["l2_rate"] = -math.log(current["l2_ratio"]) / log_n_dofs_ratio
            current["h1_rate"] = -math.log(current["h1_ratio"]) / log_n_dofs_ratio

    _print_and_save_table(rows, directory / "convergence_table.csv")
    save_plot(
        plot_convergence_rates(
            [row["n_dofs"] for row in rows],
            [row["l2_error"] for row in rows],
            [row["h1_error"] for row in rows],
        ),
        directory / "convergence_rates.png",
    )
    print("\nPlots and table saved to", directory)


if __name__ == "__main__":
    main()
