"""
Plots of solutions, errors, error estimators, and convergence rates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jax_fem.diagnostics import compute_energy_error, compute_l2_error
from jax_fem.diagnostics.trace_fluxes import evaluate_trace_fluxes
from jax_fem.function import FiniteElementFunction, evaluate_finite_element_function
from jax_fem.mesh import DfnMesh, TriangleMesh, create_fracture_frames
from jax_fem.problem import EllipticProblem
from jax_fem.space import CellBasis, FiniteElementSpace, create_cell_basis

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _import_pyplot():
    """
    Import ``matplotlib.pyplot``.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
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
    """
    Linear triangulation through every DOF of a P1 or P2 space.

    Parameters
    ----------
    space : FiniteElementSpace
        The space to triangulate.

    Returns
    -------
    x : jax.Array
        DOF x-coordinates, shape ``(dofs_test,)``.
    y : jax.Array
        DOF y-coordinates, shape ``(dofs_test,)``.
    triangles : jax.Array
        DOF indices of each plotting triangle, shape ``(K, 3)`` for P1 and
        ``(4 K, 3)`` for P2.

    Notes
    -----
    Matplotlib only draws 3-node triangles. A P2 cell is split into four
    triangles through its edge midpoints, so every DOF is used::

               v2
              /  \\
            m1----m0
           /  \\  /  \\
         v0----m2----v1

    giving ``(v0, m2, m1)``, ``(v1, m0, m2)``, ``(v2, m1, m0)``, and
    ``(m0, m1, m2)``, where local DOFs 0-2 are the vertices and 3-5 the
    midpoints ``m0, m1, m2``. P1 cells are returned unchanged.
    """
    number_of_local_dofs = space.element.number_of_local_dofs
    x = space.dof_coordinates[..., 0, 0]
    y = space.dof_coordinates[..., 0, 1]

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


def create_timestamped_output_directory(directory: str | Path, name: str) -> Path:
    """
    Create (and return) a fresh timestamped output directory.

    Factored out of ``save_plots`` so a refinement loop can create the
    directory once up front and save+close each figure immediately after
    producing it (``save_plot``), instead of accumulating every
    iteration's figures in memory until the very end -- which, for a loop
    of more than ``matplotlib.rcParams["figure.max_open_warning"]``
    (default 20) iterations, trips matplotlib's "more than 20 figures
    opened" ``RuntimeWarning``.

    Parameters
    ----------
    directory : str | Path
        Parent directory the timestamped folder is created under.
    name : str
        Name of the folder.

    Returns
    -------
    path : Path
        Path to the newly created directory. The folder name has the form
        ``name_YYMMDD_HHMM``.
    """
    timestamp = datetime.now(UTC).strftime("%y%m%d_%H%M")
    output_dir = Path(directory) / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plot(figure: Figure, path: str | Path, dpi: int = 500) -> None:
    """
    Save a single figure to ``path`` and close it.

    Closing right after saving keeps a refinement loop's peak number of
    simultaneously open figures at 1 regardless of how many iterations it
    runs, avoiding matplotlib's "more than 20 figures opened"
    ``RuntimeWarning`` (and the associated memory growth) that accumulating
    every iteration's figures in a dict until the end would otherwise
    cause.

    Parameters
    ----------
    figure : Figure
        The figure to save.
    path : str | Path
        Full output path, extension included.
    dpi : int
        DPI of the saved plot.
    """
    plt = _import_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


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
    output_dir = create_timestamped_output_directory(directory, name)
    extension = extension.lstrip(".")

    for plot_name, figure in plots.items():
        save_plot(figure, output_dir / f"{plot_name}.{extension}", dpi=dpi)

    return output_dir


def plot_convergence_rates(
    n_dofs: jax.Array,
    l2_errors: jax.Array,
    h1_errors: jax.Array,
    tail_fraction: float = 0.5,
) -> Figure:
    r"""
    Log-log convergence plot of :math:`L^2`/:math:`H^1` error vs. ``n_dofs``.

    A raw error-ratio between consecutive refinement levels/iterations is
    hard to read as a convergence *rate*: it conflates the error's actual
    decay with however much ``n_dofs`` happened to grow that step (which,
    for AMR in particular, varies iteration to iteration). This plots every
    point on log-log axes -- where a power law ``error ~ C * n_dofs^slope``
    is a straight line with slope ``slope`` -- and fits that ``slope`` by
    ordinary least squares (``jnp.polyfit`` degree 1 in log-log space) over
    the last ``tail_fraction`` of the points (sorted by ``n_dofs``), since
    early iterations on a coarse starting mesh are typically still
    pre-asymptotic and would bias a fit over every point toward a shallower
    (less negative) apparent rate.

    The fitted ``slope`` is the number directly comparable against FEM
    convergence theory's target exponent for the run at hand -- e.g. for
    :math:`H^1`, ``-1/2`` under an optimally-graded AMR mesh on a
    re-entrant-corner singularity vs. ``-1/3`` under uniform refinement on
    the same singularity (and correspondingly ``-1`` vs. ``-2/3`` for
    :math:`L^2`, by the standard duality argument).

    Parameters
    ----------
    n_dofs : jax.Array
        Degrees of freedom at each level/iteration, shape ``(n,)``,
        ``n >= 2``.
    l2_errors : jax.Array
        :math:`L^2` errors, shape ``(n,)``.
    h1_errors : jax.Array
        :math:`H^1` seminorm errors, shape ``(n,)``.
    tail_fraction : float
        Fraction of the (``n_dofs``-sorted) points, counted from the end,
        used to fit each rate; in ``(0, 1]``.

    Returns
    -------
    figure : Figure
        The figure containing the plot.
    """
    plt = _import_pyplot()

    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}.")

    n_dofs = jnp.asarray(n_dofs, dtype=float)
    l2_errors = jnp.asarray(l2_errors, dtype=float)
    h1_errors = jnp.asarray(h1_errors, dtype=float)
    if n_dofs.shape[0] < 2:
        raise ValueError(
            "plot_convergence_rates needs at least 2 points to fit a rate, "
            f"got {n_dofs.shape[0]}."
        )

    order = jnp.argsort(n_dofs)
    n_dofs, l2_errors, h1_errors = n_dofs[order], l2_errors[order], h1_errors[order]

    number_of_tail_points = max(2, round(tail_fraction * n_dofs.shape[0]))
    log_n_tail = jnp.log(n_dofs[-number_of_tail_points:])

    def _fit(errors: jax.Array) -> tuple[float, float]:
        """
        Least-squares line through the tail of a log-log error curve.

        Parameters
        ----------
        errors : jax.Array
            Errors sorted by number of DOFs, shape ``(n,)``.

        Returns
        -------
        line : tuple[float, float]
            Slope and intercept of ``log(error)`` against ``log(N_dofs)``.
        """
        log_errors_tail = jnp.log(errors[-number_of_tail_points:])
        slope, intercept = jnp.polyfit(log_n_tail, log_errors_tail, 1)
        return float(slope), float(intercept)

    l2_slope, l2_intercept = _fit(l2_errors)
    h1_slope, h1_intercept = _fit(h1_errors)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("$N_{dofs}$")
    axis.set_ylabel("error")
    axis.set_title("Convergence rate")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)

    fit_n = n_dofs[-number_of_tail_points:]
    axis.plot(
        n_dofs,
        l2_errors,
        "o-",
        color="tab:blue",
        label=f"$L^2$ error (fitted rate {l2_slope:.3f})",
    )
    axis.plot(
        fit_n,
        jnp.exp(l2_intercept) * fit_n**l2_slope,
        "--",
        color="tab:blue",
        alpha=0.5,
    )
    axis.plot(
        n_dofs,
        h1_errors,
        "s-",
        color="tab:orange",
        label=f"$H^1$ error (fitted rate {h1_slope:.3f})",
    )
    axis.plot(
        fit_n,
        jnp.exp(h1_intercept) * fit_n**h1_slope,
        "--",
        color="tab:orange",
        alpha=0.5,
    )

    axis.legend()

    return figure


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
        jnp.min(mesh.vertex_coordinates[..., 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 0]).item(),
    )
    axis_energy_error.set_ylim(
        jnp.min(mesh.vertex_coordinates[..., 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 1]).item(),
    )

    axis_energy_error.set_aspect("equal")
    axis_energy_error.set_title("Energy norm error")
    axis_energy_error.set_xlabel("x")
    axis_energy_error.set_ylabel("y")

    triangle_plot_energy_error = axis_energy_error.tripcolor(
        mesh.vertex_coordinates[..., 0, 0],
        mesh.vertex_coordinates[..., 0, 1],
        mesh.cells_to_vertices,
        facecolors=jnp.ravel(cell_energy_error),
        shading="flat",
        edgecolors="k",
        linewidth=0.1,
    )

    figure.colorbar(
        triangle_plot_energy_error, ax=axis_energy_error, fraction=0.046, pad=0.04
    )

    axis_l2_error.set_aspect("equal")
    axis_l2_error.set_xlim(
        jnp.min(mesh.vertex_coordinates[..., 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 0]).item(),
    )
    axis_l2_error.set_ylim(
        jnp.min(mesh.vertex_coordinates[..., 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 1]).item(),
    )

    axis_l2_error.set_title("$L^2$ norm error")
    axis_l2_error.set_xlabel("x")
    axis_l2_error.set_ylabel("y")

    triangle_plot_l2_error = axis_l2_error.tripcolor(
        mesh.vertex_coordinates[..., 0, 0],
        mesh.vertex_coordinates[..., 0, 1],
        mesh.cells_to_vertices,
        facecolors=jnp.ravel(cell_l2_error),
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
    """
    Plot the solution, exact solution, and error as 3D surfaces.

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

    exact = jnp.squeeze(problem.exact_solution(space.dof_coordinates), axis=-2)
    absolute_error = abs(solution.dof_values - exact)

    figure, (axis_solution, axis_exact, axis_error) = plt.subplots(
        1, 3, figsize=(20, 5), subplot_kw={"projection": "3d"}
    )

    axis_solution.set_xlim(
        jnp.min(mesh.vertex_coordinates[..., 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 0]).item(),
    )
    axis_solution.set_ylim(
        jnp.min(mesh.vertex_coordinates[..., 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 1]).item(),
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
        jnp.ravel(solution.dof_values),
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )

    axis_exact.set_xlim(
        jnp.min(mesh.vertex_coordinates[..., 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 0]).item(),
    )
    axis_exact.set_ylim(
        jnp.min(mesh.vertex_coordinates[..., 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 1]).item(),
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
        jnp.ravel(exact),
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )

    axis_error.set_xlim(
        jnp.min(mesh.vertex_coordinates[..., 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 0]).item(),
    )
    axis_error.set_ylim(
        jnp.min(mesh.vertex_coordinates[..., 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 1]).item(),
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
        jnp.ravel(absolute_error),
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
        Per-cell indicators ``eta_K``, shape ``(K, 1, 1, 1)`` (as returned by
        ``compute_residual_error_estimator``) or ``(K,)``.
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
        jnp.min(mesh.vertex_coordinates[..., 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 0]).item(),
    )
    y_min, y_max = (
        jnp.min(mesh.vertex_coordinates[..., 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 1]).item(),
    )

    for axis in (axis_estimator, axis_marked):
        axis.set_aspect("equal")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    axis_estimator.set_title(r"Residual error estimator $\eta_K$")
    triangle_plot_estimator = axis_estimator.tripcolor(
        mesh.vertex_coordinates[..., 0, 0],
        mesh.vertex_coordinates[..., 0, 1],
        mesh.cells_to_vertices,
        facecolors=jnp.ravel(cell_indicators),
        shading="flat",
        cmap="inferno",
        edgecolors="k",
        linewidth=0.1,
    )
    figure.colorbar(
        triangle_plot_estimator, ax=axis_estimator, fraction=0.046, pad=0.04
    )

    number_marked = int(jnp.sum(marked_cells))
    number_of_cells = mesh.cells_to_vertices.shape[0]
    axis_marked.set_title(
        f"Cells marked for refinement ({number_marked}/{number_of_cells})"
    )
    axis_marked.tripcolor(
        mesh.vertex_coordinates[..., 0, 0],
        mesh.vertex_coordinates[..., 0, 1],
        mesh.cells_to_vertices,
        facecolors=jnp.ravel(marked_cells).astype(float),
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
    Plot a mixed (RT0 flux, P0 scalar) solution.

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

    Notes
    -----
    The P0 scalar is drawn flat-shaded, one value per cell. The RT0 flux is
    drawn as arrows at the cell centroids (the point of the degree-1 rule),
    colored by magnitude. If ``problem`` provides ``exact_gradient``, a third
    panel shows the flux error against ``sigma = -A grad u`` at the same
    points.
    """
    plt = _import_pyplot()

    centroid_quadrature = flux_solution.space.element.reference_cell.create_quadrature(
        1
    )
    centroid_basis = create_cell_basis(flux_solution.space, centroid_quadrature)
    centroids = jnp.squeeze(centroid_basis.physical_points, axis=(-3, -2))
    flux_values = jnp.squeeze(
        evaluate_finite_element_function(flux_solution, centroid_basis).values,
        axis=(-3, -2),
    )
    flux_magnitude = jnp.linalg.norm(flux_values, axis=-1)

    show_error_panel = problem is not None and problem.exact_gradient is not None
    number_of_panels = 3 if show_error_panel else 2
    figure, axes = plt.subplots(1, number_of_panels, figsize=(7 * number_of_panels, 5))
    axis_scalar, axis_flux = axes[0], axes[1]

    x_min, x_max = (
        jnp.min(mesh.vertex_coordinates[..., 0, 0]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 0]).item(),
    )
    y_min, y_max = (
        jnp.min(mesh.vertex_coordinates[..., 0, 1]).item(),
        jnp.max(mesh.vertex_coordinates[..., 0, 1]).item(),
    )

    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    axis_scalar.set_title("$u_h$ (P0)")
    triangle_plot_scalar = axis_scalar.tripcolor(
        mesh.vertex_coordinates[..., 0, 0],
        mesh.vertex_coordinates[..., 0, 1],
        mesh.cells_to_vertices,
        facecolors=jnp.ravel(scalar_solution.dof_values),
        shading="flat",
        cmap="viridis",
        edgecolors="k",
        linewidth=0.1,
    )
    figure.colorbar(triangle_plot_scalar, ax=axis_scalar, fraction=0.046, pad=0.04)

    axis_flux.set_title(r"$\sigma_h$ (RT0), per-cell centroid")
    axis_flux.triplot(
        mesh.vertex_coordinates[..., 0, 0],
        mesh.vertex_coordinates[..., 0, 1],
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
        exact_flux = -(exact_gradient @ diffusion.mT)
        flux_h = evaluate_finite_element_function(flux_solution, centroid_basis).values
        flux_error_magnitude = jnp.linalg.norm(
            jnp.squeeze(flux_h - exact_flux, axis=(-3, -2)), axis=-1
        )

        axis_error.set_title(r"$|\sigma_h - \sigma_{exact}|$ at cell centroids")
        triangle_plot_error = axis_error.tripcolor(
            mesh.vertex_coordinates[..., 0, 0],
            mesh.vertex_coordinates[..., 0, 1],
            mesh.cells_to_vertices,
            facecolors=flux_error_magnitude,
            shading="flat",
            cmap="inferno",
            edgecolors="k",
            linewidth=0.1,
        )
        figure.colorbar(triangle_plot_error, ax=axis_error, fraction=0.046, pad=0.04)

    return figure


def plot_fracture_solutions(solution: FiniteElementFunction) -> dict[str, Figure]:
    """
    Plot a finite element function on each fracture of a network.

    Parameters
    ----------
    solution : FiniteElementFunction
        P1 or P2 function on a ``DfnMesh``.

    Returns
    -------
    figures : dict[str, Figure]
        One figure per fracture, keyed ``"fracture_<tag>"``.

    Raises
    ------
    TypeError
        If the mesh is not a ``DfnMesh``.

    Notes
    -----
    Each fracture is drawn in its own plane, with the in-plane coordinates
    ``x @ Q_i`` of the frames from ``create_fracture_frames``; the axis
    labels give the two tangent vectors. All fractures share one colour
    scale, and each fracture's traces are drawn as black lines. A P2
    function is drawn on the linear sub-triangulation through all its DOFs.
    """
    plt = _import_pyplot()
    space = solution.space
    mesh = space.mesh
    if not isinstance(mesh, DfnMesh):
        raise TypeError(
            f"plot_fracture_solutions needs a DfnMesh, got {type(mesh).__name__}."
        )

    _, _, triangles = _sub_triangulation_for_plotting(space)
    triangles = np.asarray(triangles)
    triangle_cells = np.arange(triangles.shape[0]) % mesh.cells_to_vertices.shape[0]
    triangle_fractures = np.asarray(mesh.cells_to_fractures)[triangle_cells]
    frames = np.asarray(create_fracture_frames(mesh))
    dof_coordinates = np.asarray(space.dof_coordinates)[:, 0, :]
    values = np.asarray(solution.dof_values)[:, 0]
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    trace_segments = vertices[
        np.asarray(mesh.facets_to_vertices)[np.asarray(mesh.trace_facets)]
    ]
    trace_fractures = np.asarray(mesh.traces_to_fractures)[
        np.asarray(mesh.trace_facets_to_traces)
    ]

    figures = {}
    for fracture, tag in enumerate(np.asarray(mesh.fracture_tags).tolist()):
        used_dofs, local_triangles = np.unique(
            triangles[triangle_fractures == fracture], return_inverse=True
        )
        planar_points = dof_coordinates[used_dofs] @ frames[fracture]

        figure, axis = plt.subplots(figsize=(7, 5))
        image = axis.tripcolor(
            planar_points[:, 0],
            planar_points[:, 1],
            np.reshape(local_triangles, (-1, 3)),
            values[used_dofs],
            shading="gouraud",
            vmin=values.min(),
            vmax=values.max(),
        )
        figure.colorbar(image, ax=axis, label="$u_h$")

        segments = (
            trace_segments[np.any(trace_fractures == fracture, axis=-1)]
            @ frames[fracture]
        )
        if segments.shape[0] > 0:
            separated = np.concatenate(
                (segments, np.full((segments.shape[0], 1, 2), np.nan)), axis=-2
            )
            lines = np.reshape(separated, (-1, 2))
            axis.plot(lines[:, 0], lines[:, 1], color="k", linewidth=1.5)

        first_tangent, second_tangent = frames[fracture].T
        axis.set_aspect("equal")
        axis.set_title(f"Fracture {tag}")
        axis.set_xlabel("x · ({:.2f}, {:.2f}, {:.2f})".format(*first_tangent))
        axis.set_ylabel("x · ({:.2f}, {:.2f}, {:.2f})".format(*second_tangent))
        figures[f"fracture_{tag}"] = figure

    return figures


def plot_trace_fluxes(
    solution: FiniteElementFunction,
    basis: CellBasis,
    transmissivity: jax.Array,
) -> dict[str, Figure]:
    """
    Plot the outflow of both fractures along each trace of a network.

    Parameters
    ----------
    solution : FiniteElementFunction
        P1 or P2 function on a ``DfnMesh``.
    basis : CellBasis
        Embedded basis of ``solution.space``.
    transmissivity : jax.Array
        Transmissivity tensor of each cell, shape ``(K, 1, 3, 3)``, e.g. from
        ``evaluate_fracture_transmissivity``.

    Returns
    -------
    figures : dict[str, Figure]
        One figure per trace, keyed ``"trace_<index>"`` in the order of
        ``mesh.traces_to_fractures``.

    Notes
    -----
    Each figure shows, against the arc length along the trace, the outflow
    of each of the two fractures (see ``evaluate_trace_fluxes``) and their
    sum, which the flux balance makes zero for the exact solution. The
    outflows are evaluated at the end points of every trace facet; they are
    exact there for P1 (constant per facet) and P2 (linear per facet) with a
    transmissivity constant on each cell, so a P1 outflow appears as a
    staircase. The legend gives each fracture's total outflow through the
    trace, integrated with the trapezoidal rule, which is exact for the
    same reason.
    """
    plt = _import_pyplot()
    mesh = solution.space.mesh

    facet_end_points = jnp.expand_dims(jnp.array([0.0, 1.0]), axis=(0, -2, -1))
    trace_fluxes = evaluate_trace_fluxes(
        solution, basis, transmissivity, facet_end_points
    )
    arc_lengths = np.asarray(trace_fluxes.arc_lengths)[..., 0, 0]
    outflows = np.asarray(trace_fluxes.outflows)[..., 0, :]
    order = np.argsort(arc_lengths, axis=-1)
    arc_lengths = np.take_along_axis(arc_lengths, order, axis=-1)
    outflows = np.take_along_axis(outflows, np.expand_dims(order, axis=-1), axis=-2)

    trace_of_facet = np.asarray(mesh.trace_facets_to_traces)
    trace_tags = np.asarray(mesh.fracture_tags)[np.asarray(mesh.traces_to_fractures)]

    figures = {}
    for trace, tags in enumerate(trace_tags.tolist()):
        on_trace = trace_of_facet == trace
        trace_arc_lengths = arc_lengths[on_trace]
        trace_outflows = outflows[on_trace]
        totals = np.sum(
            0.5
            * np.sum(trace_outflows, axis=-2)
            * np.diff(trace_arc_lengths, axis=-1),
            axis=0,
        )

        figure, axis = plt.subplots(figsize=(9, 4))
        for fracture, tag in enumerate(tags):
            axis.plot(
                np.ravel(trace_arc_lengths),
                np.ravel(trace_outflows[..., fracture]),
                label=f"fracture {tag} (total {totals[fracture]:.3e})",
            )
        axis.plot(
            np.ravel(trace_arc_lengths),
            np.ravel(np.sum(trace_outflows, axis=-1)),
            color="k",
            linestyle="--",
            linewidth=0.8,
            label="sum",
        )
        axis.axhline(0.0, color="gray", linewidth=0.5)
        axis.set_title(f"Trace {trace}: fractures {tags[0]} and {tags[1]}")
        axis.set_xlabel("arc length along the trace")
        axis.set_ylabel(r"outflow $-\mathbf{K}\nabla u_h \cdot \nu$")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        figures[f"trace_{trace}"] = figure

    return figures
