"""Tests for two-space (rectangular) and block saddle-point assembly."""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import (
    assemble_matrix,
    assemble_mixed_system,
    assemble_vector,
    integrate_cellwise,
)
from jax_fem.element import PiecewiseConstantTriangleP0, RaviartThomasTriangleRT0
from jax_fem.forms import elliptic_linear_form
from jax_fem.forms.mixed import elliptic_mixed_div_form, elliptic_mixed_mass_form
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _spaces_and_bases(degree: int = 4):
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    rt0 = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    p0 = PiecewiseConstantTriangleP0(reference_cell=ReferenceTriangle())
    space_sigma = create_finite_element_space(mesh, rt0)
    space_u = create_finite_element_space(mesh, p0)
    quadrature = ReferenceTriangle().create_quadrature(degree)
    basis_sigma = create_cell_basis(space_sigma, quadrature)
    basis_u = create_cell_basis(space_u, quadrature)
    return mesh, space_sigma, space_u, basis_sigma, basis_u


def _divergence_matrix(basis_u, basis_sigma):
    local_matrices = integrate_cellwise(
        elliptic_mixed_div_form(basis_u, basis_sigma, None), basis_u.physical_weights
    )
    return assemble_matrix(local_matrices, basis_u.space, basis_sigma.space)


def _zero_advection_reaction_problem(source_value: float = 1.0) -> EllipticProblem:
    return EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(source_value),
        dirichlet_conditions=(),
    )


@pytest.mark.unit
def test_two_space_matrix_is_rectangular() -> None:
    _, space_sigma, space_u, basis_sigma, basis_u = _spaces_and_bases()
    div_matrix = _divergence_matrix(basis_u, basis_sigma)
    assert div_matrix.shape == (space_u.number_of_dofs, space_sigma.number_of_dofs)


@pytest.mark.unit
def test_two_space_matrix_matches_manual_scatter() -> None:
    """The two-space assembly against a plain Python loop over cells."""
    _, space_sigma, space_u, basis_sigma, basis_u = _spaces_and_bases()
    div_matrix = _divergence_matrix(basis_u, basis_sigma)

    local_matrices = integrate_cellwise(
        elliptic_mixed_div_form(basis_u, basis_sigma, None), basis_u.physical_weights
    )

    dense_reference = jnp.zeros((space_u.number_of_dofs, space_sigma.number_of_dofs))
    for cell_index in range(space_u.cells_to_dofs.shape[0]):
        for local_i in range(1):
            for local_j in range(3):
                global_i = int(space_u.cells_to_dofs[cell_index, local_i])
                global_j = int(space_sigma.cells_to_dofs[cell_index, local_j])
                dense_reference = dense_reference.at[global_i, global_j].add(
                    local_matrices[cell_index, 0, local_i, local_j]
                )

    assert jnp.allclose(div_matrix.todense(), dense_reference, atol=1e-13)


@pytest.mark.unit
def test_assemble_mixed_system_block_shapes_and_structure() -> None:
    _, space_sigma, space_u, basis_sigma, basis_u = _spaces_and_bases()
    problem = _zero_advection_reaction_problem()

    block_matrix, rhs = assemble_mixed_system(basis_sigma, basis_u, problem)
    n_sigma, n_u = space_sigma.number_of_dofs, space_u.number_of_dofs
    total = n_sigma + n_u
    assert block_matrix.shape == (total, total)
    assert rhs.shape == (total, 1)

    dense = block_matrix.todense()
    m_block = dense[:n_sigma, :n_sigma]
    bt_block = dense[:n_sigma, n_sigma:]
    b_block = dense[n_sigma:, :n_sigma]
    zero_block = dense[n_sigma:, n_sigma:]

    # M is symmetric (a genuine mass form).
    assert jnp.allclose(m_block, m_block.T, atol=1e-13)
    # Top-right is exactly -(bottom-left)^T.
    assert jnp.allclose(bt_block, -b_block.T, atol=1e-13)
    # Bottom-right block is identically zero.
    assert jnp.allclose(zero_block, 0.0, atol=1e-14)

    # M and B match separately-assembled single/mixed forms exactly.
    coefficients_sigma = evaluate_elliptic_coefficients(
        problem, basis_sigma.physical_points
    )
    expected_m = assemble_matrix(
        integrate_cellwise(
            elliptic_mixed_mass_form(basis_sigma, basis_sigma, coefficients_sigma),
            basis_sigma.physical_weights,
        ),
        basis_sigma.space,
        basis_sigma.space,
    ).todense()
    expected_b = _divergence_matrix(basis_u, basis_sigma).todense()
    assert jnp.allclose(m_block, expected_m, atol=1e-13)
    assert jnp.allclose(b_block, expected_b, atol=1e-13)


@pytest.mark.unit
def test_assemble_mixed_system_rhs_is_zero_sigma_block_plus_load_vector() -> None:
    _, space_sigma, space_u, basis_sigma, basis_u = _spaces_and_bases()
    problem = _zero_advection_reaction_problem(source_value=3.0)

    _, rhs = assemble_mixed_system(basis_sigma, basis_u, problem)
    n_sigma = space_sigma.number_of_dofs

    assert jnp.allclose(rhs[:n_sigma], 0.0, atol=1e-14)

    coefficients_u = evaluate_elliptic_coefficients(problem, basis_u.physical_points)
    local_vectors = integrate_cellwise(
        elliptic_linear_form(basis_u, coefficients_u), basis_u.physical_weights
    )
    expected_load = assemble_vector(local_vectors, basis_u.space)
    assert jnp.allclose(rhs[n_sigma:], expected_load, atol=1e-13)


@pytest.mark.unit
@pytest.mark.integration
def test_assemble_mixed_system_is_solvable_and_gives_finite_solution() -> None:
    _, space_sigma, space_u, basis_sigma, basis_u = _spaces_and_bases()
    problem = _zero_advection_reaction_problem(source_value=1.0)

    block_matrix, rhs = assemble_mixed_system(basis_sigma, basis_u, problem)
    solution = jnp.linalg.solve(block_matrix.todense(), rhs)
    assert jnp.all(jnp.isfinite(solution))

    # A genuine linear-solve residual check, not just "didn't crash".
    residual = block_matrix @ solution - rhs
    assert float(jnp.linalg.norm(residual)) < 1e-9
