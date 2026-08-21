"""Uniform-refinement convergence study for the mixed RT0/P0 discretization.

See Section 5.4 of the implementation plan (``.context/implementation_plan.md``):

    ||sigma - sigma_h||_L2 = O(h), ||div(sigma - sigma_h)||_L2 = O(h),
    ||u - u_h||_L2 = O(h)

Uses ``create_poisson_sin_sin_problem`` (Section 5.3's scoping: only
homogeneous Dirichlet data is supported, and this solution vanishes on the
whole unit-square boundary).
"""

import math

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_mixed_system
from jax_fem.diagnostics import (
    compute_flux_divergence_l2_error,
    compute_flux_l2_error,
    compute_l2_error,
)
from jax_fem.element import PiecewiseConstantTriangleP0, RaviartThomasTriangleRT0
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import create_poisson_sin_sin_problem
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

pytestmark = [pytest.mark.convergence, pytest.mark.end_to_end, pytest.mark.slow]


def _errors_at_resolution(n: int) -> tuple[int, float, float, float]:
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
    flux_div_error = float(compute_flux_divergence_l2_error(sigma_h, basis_sigma, problem))
    u_error = float(compute_l2_error(u_h, basis_u, problem))
    cells = mesh.cells_to_vertices.shape[0]
    return cells, flux_error, flux_div_error, u_error


def test_mixed_rt0_p0_convergence_rates_and_monotonicity() -> None:
    resolutions = (4, 8, 16)
    results = [_errors_at_resolution(n) for n in resolutions]

    flux_errors = [r[1] for r in results]
    flux_div_errors = [r[2] for r in results]
    u_errors = [r[3] for r in results]

    assert all(a > b for a, b in zip(flux_errors, flux_errors[1:], strict=False))
    assert all(a > b for a, b in zip(flux_div_errors, flux_div_errors[1:], strict=False))
    assert all(a > b for a, b in zip(u_errors, u_errors[1:], strict=False))

    for (cells_coarse, flux_c, div_c, u_c), (cells_fine, flux_f, div_f, u_f) in zip(
        results, results[1:], strict=False
    ):
        h_ratio = math.sqrt(cells_fine / cells_coarse)  # h_coarse / h_fine
        flux_rate = math.log(flux_c / flux_f) / math.log(h_ratio)
        div_rate = math.log(div_c / div_f) / math.log(h_ratio)
        u_rate = math.log(u_c / u_f) / math.log(h_ratio)
        assert flux_rate == pytest.approx(1.0, abs=0.2)
        assert div_rate == pytest.approx(1.0, abs=0.2)
        assert u_rate == pytest.approx(1.0, abs=0.2)
