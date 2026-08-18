"""Tests for elliptic coefficient evaluation and the standard forms.

See Sections 12 and 13 of the architecture specification; ELL-01, ELL-02,
and ELL-03 of the FEM testing plan. ELL-04 (variable-coefficient
manufactured problem) is covered end to end in
``convergence/test_manufactured_problems.py`` (``SimpleEllipticProblem``).
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

# Using the reference triangle itself as the sole physical cell: an identity
# geometry map, so physical gradients equal the known reference gradients.
_REFERENCE_TRIANGLE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
_SINGLE_CELL = ((0, 1, 2),)
_AREA = 0.5


def _make_basis(exactness_degree: int = 2):
    mesh = create_triangle_mesh_from_arrays(
        _REFERENCE_TRIANGLE_VERTICES, _SINGLE_CELL
    )
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(exactness_degree)
    return create_cell_basis(space, quadrature)


def _integrate(form, basis, coefficients):
    pointwise = form(basis, coefficients)
    return jnp.sum(basis.physical_weights * pointwise, axis=1)


@pytest.mark.unit
def test_evaluate_elliptic_coefficients_shapes_and_values() -> None:
    basis = _make_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.array([0.3, -0.1])),
        reaction=constant_scalar_coefficient(2.0),
        source=constant_scalar_coefficient(5.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)

    k, q = basis.physical_points.shape[:2]
    assert coefficients.diffusion.shape == (k, q, 2, 2)
    assert coefficients.advection.shape == (k, q, 1, 2)
    assert coefficients.reaction.shape == (k, q, 1, 1)
    assert coefficients.source.shape == (k, q, 1, 1)

    assert jnp.allclose(coefficients.diffusion, jnp.eye(2), atol=1e-14, rtol=1e-12)
    assert jnp.allclose(coefficients.reaction, 2.0, atol=1e-14, rtol=1e-12)
    assert jnp.allclose(coefficients.source, 5.0, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
def test_evaluate_elliptic_coefficients_rejects_bad_shape() -> None:
    basis = _make_basis()

    def bad_diffusion(x):
        return jnp.ones(x.shape[:-2] + (3, 3))  # wrong: should be (d, d) = (2, 2)

    problem = EllipticProblem(
        diffusion=bad_diffusion,
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    with pytest.raises(ValueError, match="diffusion"):
        evaluate_elliptic_coefficients(problem, basis.physical_points)


@pytest.mark.unit
def test_advection_term_matches_hand_derivation() -> None:
    """ELL-03: assembly and sign convention of int v beta . grad u."""
    basis = _make_basis()
    beta = (2.0, -1.0)
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.zeros((2, 2))),
        advection=constant_vector_coefficient(jnp.asarray(beta)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    local_matrix = _integrate(elliptic_bilinear_form, basis, coefficients)[0]

    # grad phi_0 = (-1,-1), grad phi_1 = (1,0), grad phi_2 = (0,1); beta.grad_j
    # is constant, so entry (i, j) = (beta.grad_j) * int phi_i = (beta.grad_j) * area/3,
    # independent of i since every P1 hat function integrates to area/3.
    beta_dot_grad = jnp.array([-beta[0] - beta[1], beta[0], beta[1]])
    expected = jnp.outer(jnp.full(3, _AREA / 3.0), beta_dot_grad)
    assert jnp.allclose(local_matrix, expected, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
def test_reaction_global_matrix_equals_c_times_mass_matrix() -> None:
    """ELL-02: the reaction matrix is exactly c times the mass matrix."""
    basis = _make_basis()
    c = 2.5
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.zeros((2, 2))),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(c),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    reaction_matrix = _integrate(elliptic_bilinear_form, basis, coefficients)[0]

    mass_matrix = (_AREA / 12.0) * jnp.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
    )
    assert jnp.allclose(reaction_matrix, c * mass_matrix, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.end_to_end
def test_diffusion_reaction_constrained_system_is_positive_definite(
    four_triangle_center_vertex_mesh, p1_element
) -> None:
    """ELL-02: the constrained diffusion+reaction system is positive definite."""
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p1_element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(3.0),
        source=constant_scalar_coefficient(1.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(
                mesh, constant_scalar_coefficient(0.0)
            ),
        ),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    eigenvalues = jnp.linalg.eigvalsh(condensed.matrix.todense())
    assert jnp.all(eigenvalues > 0.0)


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.end_to_end
def test_anisotropic_diffusion_affine_patch_test(
    four_triangle_center_vertex_mesh, p1_element
) -> None:
    """ELL-01: off-diagonal diffusion tensor contraction, via an affine patch test.

    ``grad(v)^T A grad(u)`` must be contracted as a genuine bilinear form,
    not ``A_scalar * (grad v . grad u)`` (only valid for scalar isotropic
    diffusion, Section 13 of the testing plan): an off-diagonal ``A`` entry
    catches a swapped-index or accidental-scalar-shortcut bug. ``f`` is
    computed analytically: ``div(A grad u) == 0`` for constant ``A`` and
    affine ``u``.
    """
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p1_element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)

    a, b, c = 2.0, -3.0, 1.0

    def u_exact(x):
        return (a * x[..., 0] + b * x[..., 1] + c)[..., None]

    anisotropic_tensor = jnp.array([[2.0, 0.4], [0.4, 3.0]])
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(anisotropic_tensor),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, u_exact),
        ),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    assert jnp.allclose(matrix.todense(), matrix.todense().T, atol=1e-14, rtol=1e-12)

    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full_dof_values = expand_condensed_solution(free_dof_values, condensed)
    expected = u_exact(space.dof_coordinates)[:, :, 0]
    assert jnp.allclose(full_dof_values, expected, atol=1e-11)
