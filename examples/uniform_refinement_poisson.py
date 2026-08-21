"""Uniform-refinement convergence study, P1 Poisson on the L-shaped domain.

Solves ``create_poisson_l_shaped_singular_problem`` (a genuine re-entrant
270-degree corner singularity, Section 4.4 of the implementation plan,
``.context/implementation_plan.md``) at each level of an exact 1-to-4
uniformly refined mesh hierarchy (``create_gmsh_mesh_hierarchy``). At every
level, both the 2D and 3D error plots are saved; at the end, a convergence
table (N_dofs, mesh size h, L2/H1 errors, and their ratios between
consecutive levels) is written next to the plots and printed to the
console. Because of the corner singularity, the observed rates undershoot
the interior-regularity rates (L2 ~ h^2, H1 ~ h) -- see
``examples/adaptive_refinement_poisson.py`` for the AMR alternative that
recovers a better rate by concentrating refinement at the corner instead of
refining everywhere uniformly.

Run with:

    uv run python examples/uniform_refinement_poisson.py
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
    plot_fem_error,
    plot_fem_error_3d,
    save_plots,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import cell_diameters, create_gmsh_mesh_hierarchy, l_shaped_gmsh_geometry
from jax_fem.problem import (
    create_poisson_l_shaped_singular_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_TABLE_COLUMNS = ["level", "n_dofs", "h", "l2_error", "h1_error", "l2_ratio", "h1_ratio"]


def _solve_level(mesh, element):
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

    l2_error = float(compute_l2_error(solution, basis, problem))
    h1_error = float(compute_h1_seminorm_error(solution, basis, problem))
    h = float(jnp.mean(cell_diameters(mesh)))

    return space, basis, problem, solution, l2_error, h1_error, h


def _print_and_save_table(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'level':>5} {'N_dofs':>8} {'h':>10} {'L2 error':>12} {'H1 error':>12} "
        f"{'L2 ratio':>10} {'H1 ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        l2_ratio = row["l2_ratio"]
        h1_ratio = row["h1_ratio"]
        l2_ratio_str = "-" if math.isnan(l2_ratio) else f"{l2_ratio:.3f}"
        h1_ratio_str = "-" if math.isnan(h1_ratio) else f"{h1_ratio:.3f}"
        print(
            f"{row['level']:>5} {row['n_dofs']:>8} {row['h']:>10.5f} "
            f"{row['l2_error']:>12.6e} {row['h1_error']:>12.6e} "
            f"{l2_ratio_str:>10} {h1_ratio_str:>10}"
        )


def main(base_h: float = 0.4, levels: int = 3) -> None:
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    meshes = create_gmsh_mesh_hierarchy(l_shaped_gmsh_geometry, base_h, levels=levels)

    plots: dict = {}
    rows: list[dict] = []
    for level, mesh in enumerate(meshes):
        space, basis, problem, solution, l2_error, h1_error, h = _solve_level(
            mesh, element
        )

        plots[f"level_{level}_error_mesh"] = plot_fem_error(
            solution, basis, problem, mesh
        )
        plots[f"level_{level}_error_mesh_3d"] = plot_fem_error_3d(
            solution, basis, problem, mesh
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
            }
        )
        print(
            f"level {level}: N_dofs={space.number_of_dofs:5d}  h={h:.5f}  "
            f"L2={l2_error:.6e}  H1={h1_error:.6e}"
        )

    for previous, current in zip(rows, rows[1:], strict=False):
        current["l2_ratio"] = previous["l2_error"] / current["l2_error"]
        current["h1_ratio"] = previous["h1_error"] / current["h1_error"]

    directory = save_plots(
        plots, directory="experiments", name="uniform_refinement_poisson_l_shaped"
    )
    _print_and_save_table(rows, directory / "convergence_table.csv")
    print("\nPlots and table saved to", directory)


if __name__ == "__main__":
    main()
