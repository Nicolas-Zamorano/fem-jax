"""Uniform-refinement convergence study for the mixed RT0/P0 discretization.

Companion to ``examples/uniform_refinement_poisson.py``, for the mixed
(RT0 flux / P0 scalar) discretization of Section 5 of the implementation
plan (``.context/implementation_plan.md``). Uses
``create_poisson_sin_sin_problem`` on the unit square (Section 5.3's
scoping: only homogeneous Dirichlet data is supported, and this solution
vanishes on the whole boundary), refined uniformly via
``create_gmsh_unit_square_mesh(n)`` for doubling ``n`` (so ``h = 1/n``
halves exactly each level).

At every level, two figures are saved: ``plot_mixed_solution`` (u_h, the
flux field, and the flux error), and a second, mixed-specific plot showing
``div(sigma_h)`` against the source ``f`` -- the pointwise identity the
mixed formulation enforces exactly in its second block equation (Section
5.1). At the end, a convergence table is written with N_dofs, h, and the
three error norms this discretization actually controls (flux L2, flux
divergence L2, u L2 -- the mixed analogues of the primal method's H1/L2
pair), plus their ratios between consecutive levels. Expect all three
errors ~ O(h) (RT0/P0's known optimal rate, no singularity involved here).

Run with:

    uv run python examples/mixed_uniform_refinement.py
"""

import csv
import math
from pathlib import Path

import jax.numpy as jnp

from jax_fem.assembly import assemble_mixed_system
from jax_fem.diagnostics import (
    compute_flux_divergence_l2_error,
    compute_flux_l2_error,
    compute_l2_error,
    create_timestamped_output_directory,
    plot_mixed_solution,
    save_plot,
)
from jax_fem.element import PiecewiseConstantTriangleP0, RaviartThomasTriangleRT0
from jax_fem.function import FiniteElementFunction, evaluate_finite_element_function
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import create_poisson_sin_sin_problem
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_TABLE_COLUMNS = [
    "level",
    "n",
    "n_dofs",
    "h",
    "flux_l2_error",
    "flux_div_l2_error",
    "u_l2_error",
    "flux_l2_ratio",
    "flux_div_l2_ratio",
    "u_l2_ratio",
]


def _plot_divergence_vs_source(flux_solution, problem, mesh):
    """div(sigma_h) against the source f, at cell centroids (both P0-shaped:
    one value per cell)."""
    import matplotlib.pyplot as plt

    space = flux_solution.space
    centroid_quadrature = space.element.reference_cell.create_quadrature(1)
    basis = create_cell_basis(space, centroid_quadrature)
    divergence_h = evaluate_finite_element_function(flux_solution, basis).divergence[
        :, 0, 0, 0
    ]
    source = problem.source(basis.physical_points)[:, 0, 0, 0]

    figure, (axis_div, axis_source) = plt.subplots(1, 2, figsize=(15, 5))
    x_min, x_max = (
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    y_min, y_max = (
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )
    for axis, values, title in (
        (axis_div, divergence_h, r"$\mathrm{div}(\sigma_h)$"),
        (axis_source, source, "$f$ (source)"),
    ):
        axis.set_aspect("equal")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(title)
        plot = axis.tripcolor(
            mesh.vertex_coordinates[:, 0, 0],
            mesh.vertex_coordinates[:, 0, 1],
            mesh.cells_to_vertices,
            facecolors=values,
            shading="flat",
            cmap="viridis",
            edgecolors="k",
            linewidth=0.1,
        )
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)

    return figure


def _solve_level(n: int):
    mesh = create_gmsh_unit_square_mesh(n)
    rt0 = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    p0 = PiecewiseConstantTriangleP0(reference_cell=ReferenceTriangle())
    space_sigma = create_finite_element_space(mesh, rt0)
    space_u = create_finite_element_space(mesh, p0)

    quadrature = ReferenceTriangle().create_quadrature(4)
    basis_sigma = create_cell_basis(space_sigma, quadrature)
    basis_u = create_cell_basis(space_u, quadrature)

    problem = create_poisson_sin_sin_problem(mesh)
    block_matrix, rhs = assemble_mixed_system(basis_sigma, basis_u, problem)
    solution = jnp.linalg.solve(block_matrix.todense(), rhs)
    sigma_h = FiniteElementFunction(
        space=space_sigma, dof_values=solution[: space_sigma.number_of_dofs]
    )
    u_h = FiniteElementFunction(
        space=space_u, dof_values=solution[space_sigma.number_of_dofs :]
    )

    flux_error = float(compute_flux_l2_error(sigma_h, basis_sigma, problem))
    flux_div_error = float(
        compute_flux_divergence_l2_error(sigma_h, basis_sigma, problem)
    )
    u_error = float(compute_l2_error(u_h, basis_u, problem))
    n_dofs = space_sigma.number_of_dofs + space_u.number_of_dofs

    return mesh, sigma_h, u_h, problem, n_dofs, flux_error, flux_div_error, u_error


def _print_and_save_table(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'level':>5} {'n':>4} {'N_dofs':>8} {'h':>8} {'flux L2':>12} "
        f"{'flux div L2':>12} {'u L2':>12} {'flux ratio':>10} "
        f"{'div ratio':>10} {'u ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ratios = {
            key: ("-" if math.isnan(row[key]) else f"{row[key]:.3f}")
            for key in ("flux_l2_ratio", "flux_div_l2_ratio", "u_l2_ratio")
        }
        print(
            f"{row['level']:>5} {row['n']:>4} {row['n_dofs']:>8} {row['h']:>8.5f} "
            f"{row['flux_l2_error']:>12.6e} {row['flux_div_l2_error']:>12.6e} "
            f"{row['u_l2_error']:>12.6e} {ratios['flux_l2_ratio']:>10} "
            f"{ratios['flux_div_l2_ratio']:>10} {ratios['u_l2_ratio']:>10}"
        )


def main(resolutions: tuple[int, ...] = (4, 8, 16, 32)) -> None:
    directory = create_timestamped_output_directory(
        "experiments", "mixed_uniform_refinement_sin_sin"
    )
    rows: list[dict] = []
    for level, n in enumerate(resolutions):
        mesh, sigma_h, u_h, problem, n_dofs, flux_error, flux_div_error, u_error = (
            _solve_level(n)
        )

        save_plot(
            plot_mixed_solution(sigma_h, u_h, mesh, problem),
            directory / f"level_{level}_mixed_solution.png",
        )
        save_plot(
            _plot_divergence_vs_source(sigma_h, problem, mesh),
            directory / f"level_{level}_divergence_vs_source.png",
        )

        rows.append(
            {
                "level": level,
                "n": n,
                "n_dofs": n_dofs,
                "h": 1.0 / n,
                "flux_l2_error": flux_error,
                "flux_div_l2_error": flux_div_error,
                "u_l2_error": u_error,
                "flux_l2_ratio": float("nan"),
                "flux_div_l2_ratio": float("nan"),
                "u_l2_ratio": float("nan"),
            }
        )
        print(
            f"level {level} (n={n}): N_dofs={n_dofs:5d}  h={1.0 / n:.5f}  "
            f"flux L2={flux_error:.6e}  flux div L2={flux_div_error:.6e}  "
            f"u L2={u_error:.6e}"
        )

    for previous, current in zip(rows, rows[1:], strict=False):
        current["flux_l2_ratio"] = previous["flux_l2_error"] / current["flux_l2_error"]
        current["flux_div_l2_ratio"] = (
            previous["flux_div_l2_error"] / current["flux_div_l2_error"]
        )
        current["u_l2_ratio"] = previous["u_l2_error"] / current["u_l2_error"]

    _print_and_save_table(rows, directory / "convergence_table.csv")
    print("\nPlots and table saved to", directory)


if __name__ == "__main__":
    main()
