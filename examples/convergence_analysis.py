"""Uniform-refinement convergence study, on both a structured and a Gmsh mesh.

Same per-resolution workflow as the other examples (mesh -> element ->
space -> basis -> problem -> assembly -> Dirichlet condensation ->
user-chosen solve -> reconstruction -> error measurement), repeated across
a sequence of refinements for two independent mesh sources:
``create_structured_unit_square_mesh`` and a Gmsh-built unit square (see
``solve_with_gmsh_mesh.py``). Mesh refinement itself is out of scope for
the library (Section 2.1): each level here is a fresh mesh, not a literal
refinement of a coarser one -- for the structured source that means a new
``n x n`` grid, and for Gmsh a fresh triangulation at a smaller target
element size, not a subdivision of the previous mesh.

Uses ``create_poisson_sin_sin_problem`` (homogeneous Dirichlet BC, smooth
solution): P1 theory predicts L2 error ~ O(h^2) and H1-seminorm error ~
O(h), and this problem reaches that asymptotic regime already at the
coarsest resolution used here (unlike e.g. ``simple`` or ``exponential``,
see ``tests/convergence/test_elliptic_convergence.py``), which keeps the
resulting rate table easy to read.

Results (resolution/mesh size, cell and DOF counts, errors, and observed
convergence rates) are written to a timestamped CSV file under
``experiments/``, mirroring ``save_plots``'s convention for figures.

Requires the optional 'gmsh' dependency: pip install 'jax-fem[gmsh]'.

Run with:

    uv run python examples/convergence_analysis.py
"""

import csv
import math
from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp

# Local import: reuses the Gmsh-mesh helper defined in the other example
# script, so both convergence sources are built the same way it demonstrates.
from solve_with_gmsh_mesh import build_gmsh_unit_square_mesh  # noqa: E402

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import compute_h1_seminorm_error, compute_l2_error
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import TriangleMesh, create_structured_unit_square_mesh
from jax_fem.problem import (
    create_poisson_sin_sin_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_RESOLUTIONS = (4, 8, 16, 32, 64, 128)
_QUADRATURE_DEGREE = 4


def _solve_and_measure(mesh: TriangleMesh) -> tuple[int, int, int, float, float]:
    """One full solve; returns (cells, dofs, free_dofs, l2_error, h1_error)."""
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(_QUADRATURE_DEGREE)
    basis = create_cell_basis(space, quadrature)

    problem = create_poisson_sin_sin_problem(mesh)

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

    number_of_cells = mesh.cells_to_vertices.shape[0]
    return (
        number_of_cells,
        space.number_of_dofs,
        int(condensed.free_dofs.shape[0]),
        l2_error,
        h1_error,
    )


def _rate(
    error_coarse: float, error_fine: float, h_coarse: float, h_fine: float
) -> float:
    return math.log(error_coarse / error_fine) / math.log(h_coarse / h_fine)


def _run_study(mesh_type: str, meshes: list[tuple[float, TriangleMesh]]) -> list[dict]:
    """``meshes`` is a list of (h, mesh) pairs, coarsest first."""
    rows = []
    previous = None
    for h, mesh in meshes:
        cells, dofs, free_dofs, l2_error, h1_error = _solve_and_measure(mesh)
        if previous is None:
            l2_rate = h1_rate = None
        else:
            h_previous, l2_previous, h1_previous = previous
            l2_rate = _rate(l2_previous, l2_error, h_previous, h)
            h1_rate = _rate(h1_previous, h1_error, h_previous, h)
        rows.append(
            {
                "mesh_type": mesh_type,
                "h": h,
                "cells": cells,
                "dofs": dofs,
                "free_dofs": free_dofs,
                "l2_error": l2_error,
                "h1_error": h1_error,
                "l2_rate": l2_rate,
                "h1_rate": h1_rate,
            }
        )
        previous = (h, l2_error, h1_error)
    return rows


def _save_table_csv(rows: list[dict], directory: str | Path, name: str) -> Path:
    """Save ``rows`` as a CSV file in a timestamped folder, mirroring
    ``diagnostics.save_plots``'s naming convention for figures.
    """
    timestamp = datetime.now(UTC).strftime("%y%m%d_%H%M")
    output_dir = Path(directory) / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "convergence.csv"

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'mesh':<10}{'h':>10}{'cells':>8}{'dofs':>8}"
        f"{'l2_error':>14}{'h1_error':>14}{'l2_rate':>10}{'h1_rate':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        l2_rate = f"{row['l2_rate']:.3f}" if row["l2_rate"] is not None else "--"
        h1_rate = f"{row['h1_rate']:.3f}" if row["h1_rate"] is not None else "--"
        print(
            f"{row['mesh_type']:<10}{row['h']:>10.4f}{row['cells']:>8}{row['dofs']:>8}"
            f"{row['l2_error']:>14.6e}{row['h1_error']:>14.6e}"
            f"{l2_rate:>10}{h1_rate:>10}"
        )


def main() -> None:
    structured_meshes = [
        (1.0 / n, create_structured_unit_square_mesh(n)) for n in _RESOLUTIONS
    ]
    gmsh_meshes = [
        (1.0 / n, build_gmsh_unit_square_mesh(mesh_size=1.0 / n)) for n in _RESOLUTIONS
    ]

    rows = _run_study("structured", structured_meshes) + _run_study("gmsh", gmsh_meshes)

    _print_table(rows)

    path = _save_table_csv(rows, directory="experiments", name="convergence_analysis")
    print("\nTable saved to", path)


if __name__ == "__main__":
    main()
