"""Tests for local (single-cell) form integration.

See Sections 13 and 14.1 of the architecture specification; LOC-01 through
LOC-06 of the FEM testing plan. LOC-07 (higher-order reference matrices) is
a future-extension test, absent until a P2+ element exists.
"""

import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
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

# Using the reference triangle itself as the sole physical cell: an identity
# geometry map, so physical gradients equal the known reference gradients
# and hand-derived stiffness/mass matrices apply directly.
_REFERENCE_TRIANGLE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
_SINGLE_CELL = ((0, 1, 2),)
_AREA = 0.5

_IDENTITY = jnp.eye(2)
_ZERO_VECTOR = jnp.zeros(2)
_ZERO_TENSOR = jnp.zeros((2, 2))


def _make_basis(vertices=_REFERENCE_TRIANGLE_VERTICES, exactness_degree: int = 3):
    mesh = create_triangle_mesh_from_arrays(vertices, _SINGLE_CELL)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(exactness_degree)
    return create_cell_basis(space, quadrature)


def _local_bilinear_matrix(basis, diffusion, advection, reaction) -> jnp.ndarray:
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(diffusion),
        advection=constant_vector_coefficient(advection),
        reaction=constant_scalar_coefficient(reaction),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    pointwise = elliptic_bilinear_form(basis, coefficients)
    return jnp.sum(basis.physical_weights * pointwise, axis=1)[0]  # (N_phi, N_phi)


@pytest.mark.unit
def test_exact_p1_reference_stiffness_matrix() -> None:
    """LOC-01: local Poisson stiffness on K_hat, A = I, against a hard-coded matrix."""
    basis = _make_basis()
    stiffness = _local_bilinear_matrix(
        basis, diffusion=_IDENTITY, advection=_ZERO_VECTOR, reaction=0.0
    )
    expected = jnp.array(
        [
            [1.0, -0.5, -0.5],
            [-0.5, 0.5, 0.0],
            [-0.5, 0.0, 0.5],
        ]
    )
    assert jnp.allclose(stiffness, expected)


@pytest.mark.unit
def test_exact_p1_reference_mass_matrix() -> None:
    """LOC-02: local mass matrix on K_hat against a hard-coded matrix."""
    basis = _make_basis()
    mass = _local_bilinear_matrix(
        basis, diffusion=_ZERO_TENSOR, advection=_ZERO_VECTOR, reaction=1.0
    )
    expected = (1.0 / 24.0) * jnp.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    )
    assert jnp.allclose(mass, expected)


@pytest.mark.unit
def test_exact_p1_constant_load_vector() -> None:
    """LOC-03: local load vector on K_hat, f = 1, against a hard-coded vector."""
    basis = _make_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(_ZERO_TENSOR),
        advection=constant_vector_coefficient(_ZERO_VECTOR),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(1.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    pointwise = elliptic_linear_form(basis, coefficients)
    load_vector = jnp.sum(basis.physical_weights * pointwise, axis=1)[0]  # (3, 1)

    assert load_vector.shape == (3, 1)
    expected = (1.0 / 6.0) * jnp.ones((3, 1))
    assert jnp.allclose(load_vector, expected)


@pytest.mark.unit
@pytest.mark.property
def test_local_diffusion_matrix_properties(affine_triangle_vertices) -> None:
    """LOC-04: symmetric, positive semidefinite, K @ 1 == 0 (constant nullspace)."""
    basis = _make_basis(affine_triangle_vertices)
    stiffness = _local_bilinear_matrix(
        basis, diffusion=_IDENTITY, advection=_ZERO_VECTOR, reaction=0.0
    )

    assert jnp.allclose(stiffness, stiffness.T)
    eigenvalues = jnp.linalg.eigvalsh(stiffness)
    assert jnp.all(eigenvalues >= -1e-10)
    assert jnp.allclose(stiffness @ jnp.ones(3), 0.0, atol=1e-10)


@pytest.mark.unit
@pytest.mark.property
def test_local_mass_matrix_properties(affine_triangle_vertices) -> None:
    """LOC-05: symmetric positive definite, for any affine triangle."""
    basis = _make_basis(affine_triangle_vertices, exactness_degree=3)
    mass = _local_bilinear_matrix(
        basis, diffusion=_ZERO_TENSOR, advection=_ZERO_VECTOR, reaction=1.0
    )

    assert jnp.allclose(mass, mass.T)
    eigenvalues = jnp.linalg.eigvalsh(mass)
    assert jnp.all(eigenvalues > 0.0)


@pytest.mark.unit
@pytest.mark.property
def test_geometric_scaling() -> None:
    """LOC-06: for constant coefficients, K_sK == K_K and M_sK == s^2 * M_K."""
    scale = 2.7
    base_vertices = ((0.3, -0.2), (2.1, 0.4), (0.7, 1.8))  # a nontrivial skew triangle
    scaled_vertices = tuple((scale * x, scale * y) for x, y in base_vertices)

    def matrices(vertices):
        basis = _make_basis(vertices, exactness_degree=3)
        stiffness = _local_bilinear_matrix(
            basis, diffusion=_IDENTITY, advection=_ZERO_VECTOR, reaction=0.0
        )
        mass = _local_bilinear_matrix(
            basis, diffusion=_ZERO_TENSOR, advection=_ZERO_VECTOR, reaction=1.0
        )
        return stiffness, mass

    stiffness_base, mass_base = matrices(base_vertices)
    stiffness_scaled, mass_scaled = matrices(scaled_vertices)

    assert jnp.allclose(stiffness_scaled, stiffness_base, atol=1e-9)
    assert jnp.allclose(mass_scaled, scale**2 * mass_base, atol=1e-8)
