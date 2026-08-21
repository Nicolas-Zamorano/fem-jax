"""Tests for the mixed (H(div) flux) error diagnostics.

See Section 5.4 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_mixed_system
from jax_fem.diagnostics import (
    compute_flux_divergence_l2_error,
    compute_flux_l2_error,
    compute_l2_error,
)
from jax_fem.diagnostics.mixed_norms import (
    flux_divergence_l2_error_density,
    flux_l2_error_density,
)
from jax_fem.element import PiecewiseConstantTriangleP0, RaviartThomasTriangleRT0
from jax_fem.function import FiniteElementFunction, evaluate_finite_element_function
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_poisson_sin_sin_problem,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space


def _solve(mesh, resolution_quadrature_degree: int = 4):
    rt0 = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    p0 = PiecewiseConstantTriangleP0(reference_cell=ReferenceTriangle())
    space_sigma = create_finite_element_space(mesh, rt0)
    space_u = create_finite_element_space(mesh, p0)
    quadrature = ReferenceTriangle().create_quadrature(resolution_quadrature_degree)
    basis_sigma = create_cell_basis(space_sigma, quadrature)
    basis_u = create_cell_basis(space_u, quadrature)
    return space_sigma, space_u, basis_sigma, basis_u


@pytest.mark.unit
@pytest.mark.integration
def test_flux_errors_decrease_under_refinement() -> None:
    """A harmonic (f = 0) solution with g = 0 on the *entire* boundary would
    force the trivial u == 0 by uniqueness (Section 5.3's g = 0 scoping
    admits no nontrivial "exactly representable" zero-data case), so this
    checks monotonic decrease under refinement instead of an exact-zero
    reproduction -- the meaningful correctness signal available within that
    scoping.
    """
    problem_factory = create_poisson_sin_sin_problem
    errors = []
    for n in (4, 8):
        mesh = create_gmsh_unit_square_mesh(n)
        space_sigma, space_u, basis_sigma, basis_u = _solve(mesh)
        problem = problem_factory(mesh)
        block_matrix, rhs = assemble_mixed_system(basis_sigma, basis_u, problem)
        solution = jnp.linalg.solve(block_matrix.todense(), rhs)
        sigma_h = FiniteElementFunction(
            space=space_sigma, dof_values=solution[: space_sigma.number_of_dofs]
        )
        errors.append(float(compute_flux_l2_error(sigma_h, basis_sigma, problem)))

    assert errors[1] < errors[0]


@pytest.mark.unit
@pytest.mark.integration
def test_flux_errors_are_small_positive_and_finite_for_smooth_problem() -> None:
    mesh = create_gmsh_unit_square_mesh(6)
    space_sigma, space_u, basis_sigma, basis_u = _solve(mesh)
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

    assert 0.0 < flux_error < 1.0
    assert 0.0 < flux_div_error < 5.0
    assert 0.0 < u_error < 0.5


@pytest.mark.unit
def test_flux_l2_error_requires_exact_gradient() -> None:
    mesh = create_gmsh_unit_square_mesh(2)
    space_sigma, _, basis_sigma, _ = _solve(mesh)
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
        exact_gradient=None,
    )
    sigma_h = FiniteElementFunction(
        space=space_sigma, dof_values=jnp.zeros((space_sigma.number_of_dofs, 1))
    )
    with pytest.raises(ValueError, match="exact_gradient"):
        compute_flux_l2_error(sigma_h, basis_sigma, problem)


@pytest.mark.unit
@pytest.mark.property
def test_flux_cellwise_and_global_error_consistency() -> None:
    """ERR-02-style: sum of cellwise squared flux errors equals the square
    of the global error."""
    mesh = create_gmsh_unit_square_mesh(4)
    space_sigma, space_u, basis_sigma, basis_u = _solve(mesh)
    problem = create_poisson_sin_sin_problem(mesh)

    block_matrix, rhs = assemble_mixed_system(basis_sigma, basis_u, problem)
    solution = jnp.linalg.solve(block_matrix.todense(), rhs)
    sigma_h = FiniteElementFunction(
        space=space_sigma, dof_values=solution[: space_sigma.number_of_dofs]
    )

    from jax_fem.assembly import integrate_cellwise

    fields = {"solution": evaluate_finite_element_function(sigma_h, basis_sigma)}
    density = flux_l2_error_density(basis_sigma, fields, problem)
    cellwise_squared = integrate_cellwise(density, basis_sigma.physical_weights)
    global_error = compute_flux_l2_error(sigma_h, basis_sigma, problem)
    assert float(jnp.sum(cellwise_squared)) == pytest.approx(
        float(global_error) ** 2, rel=1e-12
    )

    div_density = flux_divergence_l2_error_density(basis_sigma, fields, problem)
    div_cellwise_squared = integrate_cellwise(div_density, basis_sigma.physical_weights)
    global_div_error = compute_flux_divergence_l2_error(sigma_h, basis_sigma, problem)
    assert float(jnp.sum(div_cellwise_squared)) == pytest.approx(
        float(global_div_error) ** 2, rel=1e-12
    )
