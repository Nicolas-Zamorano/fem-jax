"""Solve the SimpleEllipticProblem manufactured problem end to end.

Follows the typical workflow of Section 16 of the architecture
specification: mesh -> element -> space -> basis -> problem -> assembly ->
Dirichlet condensation -> user-chosen solve -> reconstruction -> error
measurement. ``SimpleEllipticProblem`` (``problem/manufactured.py``) is the
general advection-diffusion-reaction case: all of diffusion, advection, and
reaction vary in space, and the exact solution is a known closed-form
non-polynomial function, so the reported errors are genuine discretization
error, not just solver round-off.

Run with:

    uv run python examples/solve_simple_elliptic_problem.py
"""

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
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import (
    create_simple_elliptic_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space


def main(resolution: int = 16) -> None:

    mesh = create_gmsh_unit_square_mesh(resolution)

    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())

    space = create_finite_element_space(mesh, element)

    quadrature = space.element.reference_cell.create_quadrature(5)
    basis = create_cell_basis(space, quadrature)

    problem = create_simple_elliptic_problem(mesh)

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

    error_figure = plot_fem_error(solution, basis, problem, mesh)

    figure_3d = plot_fem_error_3d(solution, basis, problem, mesh)

    directory = save_plots(
        {
            "error_mesh": error_figure,
            "error_mesh_3d": figure_3d,
        },
        directory="experiments",
        name=problem.name,
    )

    number_of_cells = mesh.cells_to_vertices.shape[0]
    print(f"resolution:         {resolution} x {resolution} ({number_of_cells} cells)")
    print(
        f"degrees of freedom: {space.number_of_dofs} total, "
        f"{condensed.free_dofs.shape[0]} free"
    )
    print(f"L2 error:           {l2_error:.6e}")
    print(f"H1 seminorm error:  {h1_error:.6e}")

    print("Plots saved to ", directory)


if __name__ == "__main__":
    main()
