"""Tests for the energy bilinear form and the default advection divergence."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.assembly import assemble_matrix, integrate_cellwise
from jax_fem.diagnostics import compute_energy_error
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_energy_bilinear_form
from jax_fem.function import FiniteElementFunction, interpolate_function
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import (
    create_cell_basis,
    create_finite_element_space,
    find_boundary_dofs,
)

_DIFFUSION = jnp.array([[2.0, 0.3], [0.3, 1.0]])


def _unit_square_basis(cells_per_side: int = 4, exactness_degree: int = 2):
    ticks = np.linspace(0.0, 1.0, cells_per_side + 1)
    vertices = np.array([(x, y) for y in ticks for x in ticks])
    cells = []
    for j in range(cells_per_side):
        for i in range(cells_per_side):
            a = j * (cells_per_side + 1) + i
            diagonal = a + cells_per_side + 2
            cells += [(a, a + 1, diagonal), (a, diagonal, diagonal - 1)]
    mesh = create_triangle_mesh_from_arrays(vertices, np.array(cells))
    space = create_finite_element_space(
        mesh, LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    )
    quadrature = space.element.reference_cell.create_quadrature(exactness_degree)
    return create_cell_basis(space, quadrature)


def _assemble(form, basis, coefficients):
    local_matrices = integrate_cellwise(
        form(basis, basis, coefficients), basis.physical_weights
    )
    return assemble_matrix(local_matrices, basis.space, basis.space).todense()


def _free_dofs(basis):
    space = basis.space
    boundary_dofs = find_boundary_dofs(
        space, tuple(space.mesh.boundary_tag_names.values())
    )
    return np.setdiff1d(np.arange(space.number_of_dofs), np.asarray(boundary_dofs))


@pytest.mark.unit
def test_advection_divergence_defaults_to_zero() -> None:
    basis = _unit_square_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(_DIFFUSION),
        advection=constant_vector_coefficient(jnp.array([0.5, -0.2])),
        reaction=constant_scalar_coefficient(1.5),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    number_of_cells, number_of_points = basis.physical_points.shape[:2]
    assert coefficients.advection_divergence.shape == (
        number_of_cells,
        number_of_points,
        1,
        1,
    )
    assert jnp.all(coefficients.advection_divergence == 0.0)


@pytest.mark.unit
@pytest.mark.property
def test_energy_gram_matrix_is_symmetric_and_positive_definite_on_free_dofs() -> None:
    basis = _unit_square_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(_DIFFUSION),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    gram = _assemble(elliptic_energy_bilinear_form, basis, coefficients)

    assert jnp.allclose(gram, gram.T, atol=1e-14)
    free = _free_dofs(basis)
    eigenvalues = jnp.linalg.eigvalsh(gram[np.ix_(free, free)])
    assert jnp.all(eigenvalues > 0.0)


@pytest.mark.unit
@pytest.mark.property
def test_energy_gram_matrix_is_the_symmetric_part_on_free_dofs() -> None:
    """With constant advection and P1, integration by parts is exact on H^1_0."""
    basis = _unit_square_basis()
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(_DIFFUSION),
        advection=constant_vector_coefficient(jnp.array([0.5, -0.2])),
        reaction=constant_scalar_coefficient(1.5),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = _assemble(elliptic_bilinear_form, basis, coefficients)
    gram = _assemble(elliptic_energy_bilinear_form, basis, coefficients)

    free = np.ix_(_free_dofs(basis), _free_dofs(basis))
    symmetric_part = 0.5 * (matrix + matrix.T)
    assert jnp.allclose(symmetric_part[free], gram[free], atol=1e-13)


@pytest.mark.unit
@pytest.mark.integration
def test_energy_gram_matrix_reproduces_the_energy_error() -> None:
    """For an exact solution in V_h, e^T G e equals the squared energy error."""
    basis = _unit_square_basis()

    def exact_solution(points):
        return 1.0 + 2.0 * points[..., [0]] - points[..., [1]]

    def exact_gradient(points):
        return jnp.broadcast_to(jnp.array([2.0, -1.0]), points.shape)

    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(_DIFFUSION),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(1.5),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
        exact_solution=exact_solution,
        exact_gradient=exact_gradient,
        advection_divergence=constant_scalar_coefficient(0.4),
    )
    space = basis.space
    dof_values = jnp.expand_dims(
        jnp.sin(jnp.arange(space.number_of_dofs, dtype=float)), axis=-1
    )
    solution = FiniteElementFunction(space=space, dof_values=dof_values)
    error = dof_values - interpolate_function(exact_solution, space).dof_values

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    gram = _assemble(elliptic_energy_bilinear_form, basis, coefficients)
    discrete_energy = float((error.T @ gram @ error)[0, 0])

    assert discrete_energy == pytest.approx(
        float(compute_energy_error(solution, basis, problem)) ** 2, rel=1e-12
    )
