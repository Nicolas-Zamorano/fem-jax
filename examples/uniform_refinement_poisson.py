"""Uniform-refinement convergence study, P1 Poisson, any manufactured problem.

Solves any of ``jax_fem.problem``'s manufactured elliptic problems (except
the mixed/dual formulation ones -- see ``mixed_uniform_refinement.py`` for
those) at each level of an exact 1-to-4 uniformly refined mesh hierarchy
(``create_gmsh_mesh_hierarchy``). Every problem lives on the unit square
except ``l_shaped_singular`` (a genuine re-entrant 270-degree corner
singularity, Section 4.4 of the implementation plan,
``.context/implementation_plan.md``), which lives on the L-shaped domain --
the right base mesh factory is picked automatically from ``problem_name``.

At every level, both the 2D and 3D error plots are saved (and closed)
immediately, rather than accumulating in memory across the whole run; at
the end, a convergence table (N_dofs, mesh size h, L2/H1 errors, and their
ratios between consecutive levels) is written next to the plots and
printed to the console. For the singular problems, the observed rates
undershoot the interior-regularity rates (L2 ~ h^2, H1 ~ h) -- see
``examples/adaptive_refinement_poisson.py`` for the AMR alternative that
recovers a better rate by concentrating refinement at the singularity
instead of refining everywhere uniformly.

Run with:

    uv run python examples/uniform_refinement_poisson.py
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
    create_timestamped_output_directory,
    plot_convergence_rates,
    plot_fem_error,
    plot_fem_error_3d,
    save_plot,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import (
    TriangleMesh,
    cell_diameters,
    create_gmsh_mesh_hierarchy,
    l_shaped_gmsh_geometry,
    unit_square_gmsh_geometry,
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
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

# Every problem below lives on the unit square except "l_shaped_singular",
# which needs the genuine re-entrant-corner L-shaped domain -- see
# ``_mesh_hierarchy_for_problem``.
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
    "level",
    "n_dofs",
    "h",
    "l2_error",
    "h1_error",
    "l2_ratio",
    "h1_ratio",
    "l2_rate",
    "h1_rate",
]


def _problem_factory(problem_name: str) -> Callable[[TriangleMesh], EllipticProblem]:
    try:
        return _PROBLEM_FACTORIES[problem_name]
    except KeyError:
        raise ValueError(
            f"Unknown problem {problem_name!r}; choose one of "
            f"{sorted(_PROBLEM_FACTORIES)}."
        ) from None


def _mesh_hierarchy_for_problem(
    problem_name: str, base_h: float, levels: int
) -> list[TriangleMesh]:
    """The right base-domain mesh hierarchy for ``problem_name``.

    ``l_shaped_singular`` needs the L-shaped domain; every other problem
    lives on the unit square. ``base_h`` is a target element size either
    way -- the unit-square factory instead wants an integer cell count per
    side, so it's converted via ``round(1 / base_h)``.
    """
    if problem_name in _L_SHAPED_PROBLEM_FACTORIES:
        return create_gmsh_mesh_hierarchy(l_shaped_gmsh_geometry, base_h, levels=levels)
    number_of_cells_per_side = max(1, round(1.0 / base_h))
    return create_gmsh_mesh_hierarchy(
        unit_square_gmsh_geometry, number_of_cells_per_side, levels=levels
    )


def _solve_level(mesh, element, problem_factory):
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

    l2_error = float(compute_l2_error(solution, basis, problem))
    h1_error = float(compute_h1_seminorm_error(solution, basis, problem))
    h = float(jnp.mean(cell_diameters(mesh)))

    return space, basis, problem, solution, l2_error, h1_error, h


def _fmt(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.3f}"


def _print_and_save_table(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'level':>5} {'N_dofs':>8} {'h':>10} {'L2 error':>12} {'H1 error':>12} "
        f"{'L2 ratio':>10} {'H1 ratio':>10} {'L2 rate':>10} {'H1 rate':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['level']:>5} {row['n_dofs']:>8} {row['h']:>10.5f} "
            f"{row['l2_error']:>12.6e} {row['h1_error']:>12.6e} "
            f"{_fmt(row['l2_ratio']):>10} {_fmt(row['h1_ratio']):>10} "
            f"{_fmt(row['l2_rate']):>10} {_fmt(row['h1_rate']):>10}"
        )


def main(
    problem_name: str = "l_shaped_singular",
    base_h: float = 0.4,
    levels: int = 3,
) -> None:
    problem_factory = _problem_factory(problem_name)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    meshes = _mesh_hierarchy_for_problem(problem_name, base_h, levels)

    directory = create_timestamped_output_directory(
        "experiments", f"uniform_refinement_poisson_{problem_name}"
    )
    rows: list[dict] = []
    for level, mesh in enumerate(meshes):
        space, basis, problem, solution, l2_error, h1_error, h = _solve_level(
            mesh, element, problem_factory
        )

        save_plot(
            plot_fem_error(solution, basis, problem, mesh),
            directory / f"level_{level}_error_mesh.png",
        )
        save_plot(
            plot_fem_error_3d(solution, basis, problem, mesh),
            directory / f"level_{level}_error_mesh_3d.png",
        )

        rows.append(
            {
                "level": level,
                "n_dofs": space.number_of_dofs,
                "h": h,
                "l2_error": l2_error,
                "h1_error": h1_error,
                "l2_ratio": float("nan"),
                "h1_ratio": float("nan"),
                "l2_rate": float("nan"),
                "h1_rate": float("nan"),
            }
        )
        print(
            f"level {level}: N_dofs={space.number_of_dofs:5d}  h={h:.5f}  "
            f"L2={l2_error:.6e}  H1={h1_error:.6e}"
        )

    for previous, current in zip(rows, rows[1:], strict=False):
        current["l2_ratio"] = previous["l2_error"] / current["l2_error"]
        current["h1_ratio"] = previous["h1_error"] / current["h1_error"]
        # The observed local convergence rate between consecutive levels:
        # the exponent p such that error ~ n_dofs^p, directly comparable
        # against theory's target rate (e.g. -1/3 for H1 under uniform
        # refinement of this domain's re-entrant-corner singularity),
        # unlike the raw ratio above, which conflates the error's decay
        # with however much n_dofs grew at that level (a fixed 4x here,
        # but this keeps the same convention as the AMR script).
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
