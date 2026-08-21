"""
Plotting Module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax_fem.diagnostics import compute_energy_error, compute_l2_error
from jax_fem.function import FiniteElementFunction, evaluate_finite_element_function
from jax_fem.mesh import TriangleMesh
from jax_fem.problem import EllipticProblem
from jax_fem.space import CellBasis, FiniteElementSpace, create_cell_basis

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _import_pyplot():
    """
    Import ``matplotlib.pyplot``, raising a friendly error if unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting requires the optional 'matplotlib' dependency. "
            "Install it with: pip install 'jax-fem[plotting]'."
        ) from error
    return plt


def _sub_triangulation_for_plotting(
    space: FiniteElementSpace,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build a flat, 3-node-triangle triangulation covering every DOF exactly.

    Matplotlib's ``tripcolor``/``plot_trisurf`` only understand linear
    (3-node) triangles, but a P2 ``FiniteElementFunction`` has 6 DOFs per
    cell (3 vertices + 3 edge midpoints) and genuine quadratic curvature
    between them. Plotting only the 3 corner DOFs would both discard the
    edge-midpoint DOFs entirely and silently misrepresent a quadratic
    surface as flat. Instead, each cell is split into 4 linear
    sub-triangles using the edge midpoints -- an exact (not approximate)
    piecewise-linear representation of the quadratic surface, using every
    DOF and introducing no interpolation error of its own::

               v2
              /  \\
            m1----m0
           /  \\  /  \\
         v0----m2----v1

    Sub-triangle 0: (v0, m2, m1)  Sub-triangle 1: (v1, m0, m2)
    Sub-triangle 2: (v2, m1, m0)  Sub-triangle 3 (center): (m0, m1, m2)

    where local DOF 0/1/2 are the vertices and 3/4/5 are the facet
    midpoints m0/m1/m2 (``LagrangeTriangleP2.entity_dofs``). For P1 (3
    DOFs/cell, already a flat triangle), this is the identity: the space's
    own cells are returned unchanged.

    Parameters
    ----------
    space : FiniteElementSpace
        The finite element space to build a plotting triangulation for.

    Returns
    -------
    x : jax.Array
        DOF x-coordinates, shape ``(N_dofs,)``.
    y : jax.Array
        DOF y-coordinates, shape ``(N_dofs,)``.
    triangles : jax.Array
        Sub-triangle DOF-index connectivity: shape ``(K, 3)`` for a 3-DOF
        (P1) element, ``(4 * K, 3)`` for a 6-DOF (P2) element.
    """
    number_of_local_dofs = space.element.number_of_local_dofs
    x = space.dof_coordinates[:, 0, 0]
    y = space.dof_coordinates[:, 0, 1]

    if number_of_local_dofs == 3:
        return x, y, space.cells_to_dofs

    if number_of_local_dofs == 6:
        v0, v1, v2, m0, m1, m2 = (space.cells_to_dofs[:, i] for i in range(6))
        triangles = jnp.concatenate(
            (
                jnp.stack((v0, m2, m1), axis=-1),
                jnp.stack((v1, m0, m2), axis=-1),
                jnp.stack((v2, m1, m0), axis=-1),
                jnp.stack((m0, m1, m2), axis=-1),
            ),
            axis=0,
        )
        return x, y, triangles

    raise NotImplementedError(
        "Plotting currently supports only 3-DOF (P1) and 6-DOF (P2) scalar "
        f"Lagrange elements; got an element with {number_of_local_dofs} "
        "local DOFs."
    )


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
    plt = _import_pyplot()

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

    Renders every DOF, not just mesh vertices: for a P2 solution this uses
    an exact linear sub-triangulation through the edge-midpoint DOFs too
    (Section 3 of the implementation plan), so the plotted surface shows
    the solution's real quadratic curvature rather than a flattened
    3-vertex-per-cell approximation (see ``_sub_triangulation_for_plotting``).

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
    plt = _import_pyplot()

    space = solution.space
    x, y, triangles = _sub_triangulation_for_plotting(space)

    exact = problem.exact_solution(space.dof_coordinates).reshape(-1, 1)
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
        x,
        y,
        triangles,
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
        x,
        y,
        triangles,
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
        x,
        y,
        triangles,
        absolute_error.reshape(-1),
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )

    return figure


def plot_error_estimator_and_marked_cells(
    cell_indicators: jax.Array,
    marked_cells: jax.Array,
    mesh: TriangleMesh,
) -> Figure:
    """
    Plot the residual error estimator ``eta_K`` and the cells marked for AMR.

    Parameters
    ----------
    cell_indicators : jax.Array
        Per-cell indicators ``eta_K``, shape ``(K,)`` (from
        ``compute_residual_error_estimator``).
    marked_cells : jax.Array
        Boolean mask, shape ``(K,)`` (from
        ``jax_fem.mesh.mark_cells_by_dorfler_bulk_criterion``).
    mesh : TriangleMesh
        The mesh ``cell_indicators``/``marked_cells`` were computed on.

    Returns
    -------
    figure : Figure
        The figure containing the plots.
    """
    plt = _import_pyplot()

    figure, (axis_estimator, axis_marked) = plt.subplots(1, 2, figsize=(15, 5))

    x_min, x_max = (
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    y_min, y_max = (
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )

    for axis in (axis_estimator, axis_marked):
        axis.set_aspect("equal")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    axis_estimator.set_title(r"Residual error estimator $\eta_K$")
    triangle_plot_estimator = axis_estimator.tripcolor(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        facecolors=jnp.asarray(cell_indicators).reshape(-1),
        shading="flat",
        cmap="inferno",
        edgecolors="k",
        linewidth=0.1,
    )
    figure.colorbar(
        triangle_plot_estimator, ax=axis_estimator, fraction=0.046, pad=0.04
    )

    number_marked = int(jnp.sum(marked_cells))
    axis_marked.set_title(
        f"Cells marked for refinement ({number_marked}/{mesh.cells_to_vertices.shape[0]})"
    )
    axis_marked.tripcolor(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        facecolors=jnp.asarray(marked_cells).astype(float).reshape(-1),
        shading="flat",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        edgecolors="k",
        linewidth=0.1,
    )

    return figure


def plot_mixed_solution(
    flux_solution: FiniteElementFunction,
    scalar_solution: FiniteElementFunction,
    mesh: TriangleMesh,
    problem: EllipticProblem | None = None,
) -> Figure:
    """
    Plot a mixed (RT0 flux / P0 scalar) solution.

    ``scalar_solution`` (P0) has exactly one DOF per cell, so it plots
    directly as a flat-shaded ``tripcolor`` -- no sub-triangulation needed
    (unlike ``plot_fem_error_3d``'s P1/P2 handling). ``flux_solution`` (RT0)
    is genuinely vector-valued, so it is shown as a per-cell quiver at cell
    centroids (a degree-1 quadrature rule is exactly the reference
    triangle's centroid, Section 5.2's ``CellBasis``), colored by magnitude.
    If ``problem`` is given and provides ``exact_gradient``, a third panel
    shows the pointwise flux magnitude error against
    ``sigma_exact = -A grad(u_exact)`` (Section 5.1) at the same centroids.

    Parameters
    ----------
    flux_solution : FiniteElementFunction
        The mixed finite element flux solution (e.g. RT0's ``sigma``).
    scalar_solution : FiniteElementFunction
        The mixed finite element scalar solution (e.g. P0's ``u``).
    mesh : TriangleMesh
        The mesh to plot on.
    problem : EllipticProblem | None
        The elliptic problem, for the optional exact-flux-error panel.

    Returns
    -------
    figure : Figure
        The figure containing the plots.
    """
    plt = _import_pyplot()

    centroid_quadrature = flux_solution.space.element.reference_cell.create_quadrature(1)
    centroid_basis = create_cell_basis(flux_solution.space, centroid_quadrature)
    centroids = centroid_basis.physical_points[:, 0, 0, :]  # (K, 2)
    flux_values = evaluate_finite_element_function(flux_solution, centroid_basis).values[
        :, 0, 0, :
    ]  # (K, 2)
    flux_magnitude = jnp.linalg.norm(flux_values, axis=-1)

    show_error_panel = problem is not None and problem.exact_gradient is not None
    number_of_panels = 3 if show_error_panel else 2
    figure, axes = plt.subplots(1, number_of_panels, figsize=(7 * number_of_panels, 5))
    axis_scalar, axis_flux = axes[0], axes[1]

    x_min, x_max = (
        jnp.min(mesh.vertex_coordinates[:, 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 0]).item(),
    )
    y_min, y_max = (
        jnp.min(mesh.vertex_coordinates[:, 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[:, 0, 1]).item(),
    )

    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    axis_scalar.set_title("$u_h$ (P0)")
    triangle_plot_scalar = axis_scalar.tripcolor(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        facecolors=scalar_solution.dof_values.reshape(-1),
        shading="flat",
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )
    figure.colorbar(triangle_plot_scalar, ax=axis_scalar, fraction=0.046, pad=0.04)

    axis_flux.set_title(r"$\sigma_h$ (RT0), per-cell centroid")
    axis_flux.triplot(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        color="k",
        linewidth=0.1,
    )
    quiver_plot = axis_flux.quiver(
        centroids[:, 0],
        centroids[:, 1],
        flux_values[:, 0],
        flux_values[:, 1],
        flux_magnitude,
        cmap="viridis",
    )
    figure.colorbar(quiver_plot, ax=axis_flux, fraction=0.046, pad=0.04)

    if show_error_panel:
        axis_error = axes[2]
        exact_gradient = problem.exact_gradient(centroid_basis.physical_points)
        diffusion = problem.diffusion(centroid_basis.physical_points)
        exact_flux = -(exact_gradient @ diffusion.mT)  # (K, 1, 1, d)
        flux_h = evaluate_finite_element_function(flux_solution, centroid_basis).values
        flux_error_magnitude = jnp.linalg.norm(
            (flux_h - exact_flux)[:, 0, 0, :], axis=-1
        )

        axis_error.set_title(r"$|\sigma_h - \sigma_{exact}|$ at cell centroids")
        triangle_plot_error = axis_error.tripcolor(
            mesh.vertex_coordinates[:, 0, 0],
            mesh.vertex_coordinates[:, 0, 1],
            mesh.cells_to_vertices,
            facecolors=flux_error_magnitude,
            shading="flat",
            cmap="inferno",
            edgecolors="k",
            linewidth=0.1,
        )
        figure.colorbar(triangle_plot_error, ax=axis_error, fraction=0.046, pad=0.04)

    return figure
