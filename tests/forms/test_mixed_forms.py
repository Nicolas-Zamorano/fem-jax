"""Tests for the mixed (H(div) x L^2) forms.

See Section 5.3 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax.numpy as jnp
import pytest

from jax_fem.element import PiecewiseConstantTriangleP0, RaviartThomasTriangleRT0
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

_REFERENCE_TRIANGLE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
_SINGLE_CELL = ((0, 1, 2),)


def _spaces_and_bases(vertices=_REFERENCE_TRIANGLE_VERTICES, cells=_SINGLE_CELL, degree=4):
    mesh = create_triangle_mesh_from_arrays(vertices, cells)
    rt0 = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    p0 = PiecewiseConstantTriangleP0(reference_cell=ReferenceTriangle())
    space_sigma = create_finite_element_space(mesh, rt0)
    space_u = create_finite_element_space(mesh, p0)
    quadrature = ReferenceTriangle().create_quadrature(degree)
    basis_sigma = create_cell_basis(space_sigma, quadrature)
    basis_u = create_cell_basis(space_u, quadrature)
    return mesh, basis_sigma, basis_u


@pytest.mark.unit
def test_mixed_mass_form_is_gram_matrix_for_identity_diffusion() -> None:
    """With A = I, A^-1 = I, so the mass form is exactly the pointwise Gram
    matrix ``psi_i . psi_j`` of the physical RT0 basis vectors."""
    _, basis_sigma, _ = _spaces_and_bases()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis_sigma.physical_points)
    local_matrix = elliptic_mixed_mass_form(basis_sigma, coefficients)
    expected = basis_sigma.values @ basis_sigma.values.mT

    assert local_matrix.shape == basis_sigma.values.shape[:2] + (3, 3)
    assert jnp.allclose(local_matrix, expected, atol=1e-14)
    # A Gram matrix is symmetric and positive semidefinite at every point.
    assert jnp.allclose(local_matrix, jnp.swapaxes(local_matrix, -1, -2), atol=1e-14)
    eigenvalues = jnp.linalg.eigvalsh(local_matrix[0, 0])
    assert jnp.all(eigenvalues >= -1e-12)


@pytest.mark.unit
def test_mixed_mass_form_scales_with_diffusion_inverse() -> None:
    """With A = c*I, A^-1 = (1/c)*I, so the mass form scales by 1/c."""
    _, basis_sigma, _ = _spaces_and_bases()

    def matrix_for(scale: float) -> jnp.ndarray:
        problem = EllipticProblem(
            diffusion=constant_tensor_coefficient(scale * jnp.eye(2)),
            advection=constant_vector_coefficient(jnp.zeros(2)),
            reaction=constant_scalar_coefficient(0.0),
            source=constant_scalar_coefficient(0.0),
            dirichlet_conditions=(),
        )
        coefficients = evaluate_elliptic_coefficients(
            problem, basis_sigma.physical_points
        )
        return elliptic_mixed_mass_form(basis_sigma, coefficients)

    base = matrix_for(1.0)
    scaled = matrix_for(2.0)
    assert jnp.allclose(scaled, base / 2.0, atol=1e-13)


@pytest.mark.unit
def test_mixed_div_form_matches_hand_derivation() -> None:
    """P0's single basis value is identically 1, so the div form reduces
    exactly to RT0's own physical divergence."""
    _, basis_sigma, basis_u = _spaces_and_bases()
    local_matrix = elliptic_mixed_div_form(basis_u, basis_sigma)

    assert local_matrix.shape[-2:] == (1, 3)
    expected = basis_sigma.divergences.mT  # (K, Qsigma, 1, 3), P0's value == 1
    assert jnp.allclose(local_matrix, jnp.broadcast_to(expected, local_matrix.shape), atol=1e-13)


@pytest.mark.unit
def test_mixed_div_form_is_rectangular_for_multi_cell_mesh() -> None:
    _UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    _UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))
    _, basis_sigma, basis_u = _spaces_and_bases(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    local_matrix = elliptic_mixed_div_form(basis_u, basis_sigma)
    k = 2
    assert local_matrix.shape[0] == k
    assert local_matrix.shape[-2:] == (1, 3)
