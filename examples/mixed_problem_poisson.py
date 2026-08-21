"""Solve the mixed (RT0 flux / P0 scalar) Poisson problem end to end.

Follows Section 5 of the implementation plan
(``.context/implementation_plan.md``): the mixed first-order system
``[[M, -B^T], [B, 0]] [sigma; u] = [0; F]`` is assembled directly as one
block saddle-point system (``assemble_mixed_system``) and solved with a
single dense linear solve -- no Schur-complement elimination. Uses
``create_poisson_sin_sin_problem`` (Section 5.3's scoping: only homogeneous
Dirichlet data is supported, and this solution vanishes on the whole
boundary).

Run with:

    uv run python examples/mixed_problem_poisson.py
"""

import jax.numpy as jnp

from jax_fem.assembly import assemble_mixed_system
from jax_fem.diagnostics import (
    compute_flux_divergence_l2_error,
    compute_flux_l2_error,
    compute_l2_error,
    plot_mixed_solution,
    save_plots,
)
from jax_fem.element import PiecewiseConstantTriangleP0, RaviartThomasTriangleRT0
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import create_poisson_sin_sin_problem
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space


def main(resolution: int = 12) -> None:
    mesh = create_gmsh_unit_square_mesh(resolution)

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
    flux_div_error = float(compute_flux_divergence_l2_error(sigma_h, basis_sigma, problem))
    u_error = float(compute_l2_error(u_h, basis_u, problem))

    figure = plot_mixed_solution(sigma_h, u_h, mesh, problem)
    directory = save_plots(
        {"mixed_solution": figure}, directory="experiments", name=problem.name
    )

    number_of_cells = mesh.cells_to_vertices.shape[0]
    print(f"resolution:            {resolution} x {resolution} ({number_of_cells} cells)")
    print(
        f"degrees of freedom:     {space_sigma.number_of_dofs} (sigma, RT0) + "
        f"{space_u.number_of_dofs} (u, P0) = "
        f"{space_sigma.number_of_dofs + space_u.number_of_dofs} total"
    )
    print(f"flux L2 error:          {flux_error:.6e}")
    print(f"flux divergence error:  {flux_div_error:.6e}")
    print(f"u L2 error:             {u_error:.6e}")
    print("\nPlot saved to", directory)


if __name__ == "__main__":
    main()
