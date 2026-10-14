"""Solve the two-fracture benchmark on a discrete fracture network.

Two orthogonal fractures, ``F1 = (-1, 1) x (0, 1) x {0}`` and
``F2 = {0} x (0, 1) x (-1, 1)``, cross along the trace ``x = z = 0``. The
mesh shares the trace vertices, so the standard workflow (forms ->
``integrate_cellwise`` -> assembly -> Dirichlet condensation) couples the
fractures with no extra trace terms. The only DFN-specific steps are the
mesh and ``create_embedded_cell_basis``, whose gradients are 3D vectors
tangent to each fracture.

For each element, the script plots the solution on each fracture and the
outflow of both fractures along the trace; the exact outflows are
``+2 y (1 - y)`` from ``F1`` and ``-2 y (1 - y)`` from ``F2``, with totals
``+1/3`` and ``-1/3``. The plots are saved under ``experiments/``.

A mesh written by dfnWorks can be used instead with
``read_dfn_mesh_avs("output/full_mesh.inp")``; dfnWorks centres the domain
at the origin, so the benchmark network there spans ``y`` in
``(-1/2, 1/2)``.

Run with:

    uv run python examples/dfn_two_fractures.py
"""

import jax.numpy as jnp

from jax_fem.assembly import assemble_matrix, assemble_vector, integrate_cellwise
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import (
    compute_h1_seminorm_error,
    compute_l2_error,
    plot_fracture_solutions,
    plot_trace_fluxes,
    save_plots,
)
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_dfn_two_fractures_mesh
from jax_fem.problem import (
    create_dfn_two_fractures_problem,
    evaluate_elliptic_coefficients,
    evaluate_fracture_transmissivity,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_embedded_cell_basis, create_finite_element_space


def main(cells_per_unit_length: int = 8) -> None:

    mesh = create_dfn_two_fractures_mesh(cells_per_unit_length)
    problem = create_dfn_two_fractures_problem(mesh)
    transmissivity = evaluate_fracture_transmissivity(mesh, jnp.ones(2))

    number_of_cells = mesh.cells_to_vertices.shape[0]
    print(
        f"mesh: {number_of_cells} cells, {mesh.fracture_tags.shape[0]} fractures, "
        f"{mesh.trace_facets.shape[0]} trace facets"
    )

    plots = {}
    for element in (
        LagrangeTriangleP1(reference_cell=ReferenceTriangle()),
        LagrangeTriangleP2(reference_cell=ReferenceTriangle()),
    ):
        space = create_finite_element_space(mesh, element)
        quadrature = space.element.reference_cell.create_quadrature(5)
        basis = create_embedded_cell_basis(space, quadrature)

        coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
        local_matrices = integrate_cellwise(
            elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
        )
        matrix = assemble_matrix(local_matrices, basis.space, basis.space)
        local_vectors = integrate_cellwise(
            elliptic_linear_form(basis, coefficients), basis.physical_weights
        )
        vector = assemble_vector(local_vectors, basis.space)

        dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
        condensed = condense_dirichlet_system(
            matrix, vector, dirichlet_dofs, dirichlet_values
        )

        free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)

        full_dof_values = expand_condensed_solution(free_dof_values, condensed)
        solution = FiniteElementFunction(space=space, dof_values=full_dof_values)

        l2_error = float(compute_l2_error(solution, basis, problem))
        h1_error = float(compute_h1_seminorm_error(solution, basis, problem))

        print(
            f"P{element.polynomial_degree}: {space.number_of_dofs} DOFs "
            f"({condensed.free_dofs.shape[0]} free), L2 error {l2_error:.3e}, "
            f"H1 seminorm error {h1_error:.3e}"
        )

        figures = {
            **plot_fracture_solutions(solution),
            **plot_trace_fluxes(solution, basis, transmissivity),
        }
        degree = element.polynomial_degree
        plots |= {f"p{degree}_{name}": figure for name, figure in figures.items()}

    directory = save_plots(plots, directory="experiments", name=problem.name)
    print("Plots saved to", directory)


if __name__ == "__main__":
    main()
