"""
Plotting Module.
"""

from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from jax_fem.diagnostics import compute_energy_error, compute_l2_error
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import TriangleMesh
from jax_fem.problem import EllipticProblem
from jax_fem.space import CellBasis


def save_plots(
    plots: dict[str, Figure],
    directory: str | Path,
    name: str,
    extension: str = "png",
    dpi: int = 500,
) -> Path:
    """
    Save plots in a timestamped folder.

    Parameters
    ----------
    plots : dict[str, Figure]
        Dictionary of plots to save.
    name : str
        Name of the folder.
    extension : str
        Extension of the plots.
    dpi : int
        DPI of the plots.

    Returns
    -------
    path : Path
        Path to the directory where the plots are saved.
        The folder name has the form ``name_YY-MM-DD_HH-MM``.
    """
    timestamp = datetime.now(UTC).strftime("%y%m%d_%H%M")
    output_dir = Path(directory) / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = extension.lstrip(".")

    # Save figures
    for plot_name, figure in plots.items():
        figure.savefig(
            output_dir / f"{plot_name}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )

    return output_dir


def plot_fem_error(
    solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    mesh: TriangleMesh,
) -> Figure:
    """plot ``L^2`` and energy error on the mesh.

    Parameters
    ----------
    solution : FiniteElementFunction
        The solution to plot.
    basis : CellBasis
        The basis of the solution.
    problem : EllipticProblem
        The problem to solve.
    mesh : TriangleMesh
        The mesh to plot on.

    Returns
    -------
    figure : Figure
        The figure containing the plots.
    """
    cell_energy_error = compute_energy_error(solution, basis, problem, reduce=False)
    cell_l2_error = compute_l2_error(solution, basis, problem, reduce=False)

    figure, (axis_energy_error, axis_l2_error) = plt.subplots(1, 2, figsize=(15, 5))

    axis_energy_error.set_xlim(
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    axis_energy_error.set_ylim(
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )

    axis_energy_error.set_aspect("equal")
    axis_energy_error.set_title("Energy norm error")
    axis_energy_error.set_xlabel("x")
    axis_energy_error.set_ylabel("y")

    triangle_plot_energy_error = axis_energy_error.tripcolor(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        facecolors=cell_energy_error.reshape(-1),
        shading="flat",
        edgecolors="k",
        linewidth=0.1,
    )

    figure.colorbar(
        triangle_plot_energy_error, ax=axis_energy_error, fraction=0.046, pad=0.04
    )

    axis_l2_error.set_aspect("equal")
    axis_l2_error.set_xlim(
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    axis_l2_error.set_ylim(
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )

    axis_l2_error.set_title("$L^2$ norm error")
    axis_l2_error.set_xlabel("x")
    axis_l2_error.set_ylabel("y")

    triangle_plot_l2_error = axis_l2_error.tripcolor(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        facecolors=cell_l2_error.reshape(-1),
        shading="flat",
        edgecolors="k",
        linewidth=0.1,
    )

    figure.colorbar(triangle_plot_l2_error, ax=axis_l2_error, fraction=0.046, pad=0.04)

    return figure


def plot_fem_error_3d(
    solution: FiniteElementFunction,
    basis: CellBasis,
    problem: EllipticProblem,
    mesh: TriangleMesh,
) -> Figure:
    """plot ``L^2`` and energy error on a 3D mesh.

    Parameters
    ----------
    solution : FiniteElementFunction
        The solution to plot.
    basis : CellBasis
        The basis of the solution.
    problem : EllipticProblem
        The problem to solve.
    mesh : TriangleMesh
        The mesh to plot on.

    Returns
    -------
    figure : Figure
        The figure containing the plots.
    """
    exact = problem.exact_solution(mesh.vertex_coordinates[:, 0, :])

    absolute_error = abs(solution.dof_values - exact)

    figure, (axis_solution, axis_exact, axis_error) = plt.subplots(
        1, 3, figsize=(20, 5), subplot_kw={"projection": "3d"}
    )

    axis_solution.set_xlim(
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    axis_solution.set_ylim(
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )

    axis_solution.set_zlim(
        jnp.min(exact).item(),
        jnp.max(exact).item(),
    )

    axis_solution.set_title("FEM solution")
    axis_solution.set_xlabel("x")
    axis_solution.set_ylabel("y")

    axis_solution.plot_trisurf(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        solution.dof_values.reshape(-1),
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )

    axis_exact.set_xlim(
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    axis_exact.set_ylim(
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )

    axis_exact.set_zlim(
        jnp.min(exact).item(),
        jnp.max(exact).item(),
    )

    axis_exact.set_title("Exact solution")
    axis_exact.set_xlabel("x")
    axis_exact.set_ylabel("y")

    axis_exact.plot_trisurf(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        exact.reshape(-1),
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )

    axis_error.set_xlim(
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    axis_error.set_ylim(
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )

    axis_error.set_zlim(
        0.0,
        jnp.max(absolute_error).item(),
    )

    axis_error.set_title("Absolute error")
    axis_error.set_xlabel("x")
    axis_error.set_ylabel("y")

    axis_error.plot_trisurf(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        absolute_error.reshape(-1),
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )

    return figure
