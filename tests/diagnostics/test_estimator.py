"""Tests for the residual-based a posteriori error estimator."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.assembly import (
    assemble_matrix,
    assemble_vector,
    gather_vector,
    integrate_cellwise,
)
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import (
    compute_facet_jump_indicators,
    compute_h1_seminorm_error,
    compute_residual_error_estimator,
    evaluate_facet_fluxes,
    evaluate_flux_divergence,
)
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction, interpolate_function
from jax_fem.mesh import create_gmsh_unit_square_mesh, create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
    create_poisson_sin_sin_problem,
    evaluate_elliptic_coefficients,
    isotropic_tensor_coefficient,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from jax_fem.space.facet_basis import create_facet_basis

_ZERO_PROBLEM_KWARGS = dict(
    diffusion=constant_tensor_coefficient(jnp.eye(2)),
    advection=constant_vector_coefficient(jnp.zeros(2)),
    reaction=constant_scalar_coefficient(0.0),
    source=constant_scalar_coefficient(0.0),
)


@pytest.mark.unit
@pytest.mark.integration
def test_estimator_vanishes_for_exact_affine_p1_solution(
    four_triangle_center_vertex_mesh,
) -> None:
    """A P1 discrete solution of an affine-exact problem is the true
    solution everywhere: both the interior residual and every facet jump
    are exactly zero, so eta_K == 0 for every cell.
    """
    mesh = four_triangle_center_vertex_mesh
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)

    def u_exact(x):
        return (2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0)[..., None]

    problem = EllipticProblem(
        **_ZERO_PROBLEM_KWARGS,
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, u_exact),
        ),
    )
    quadrature = space.element.reference_cell.create_quadrature(3)
    basis = create_cell_basis(space, quadrature)
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
    free = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full = expand_condensed_solution(free, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full)

    facet_quadrature = ReferenceInterval().create_quadrature(3)
    facet_basis = create_facet_basis(basis, facet_quadrature)
    cell_indicators, global_estimator = compute_residual_error_estimator(
        solution, basis, facet_basis, problem
    )
    assert jnp.max(jnp.abs(cell_indicators)) < 1e-10
    assert abs(float(global_estimator)) < 1e-10


def _structured_unit_square(cells_per_side: int = 4):
    ticks = np.linspace(0.0, 1.0, cells_per_side + 1)
    vertices = np.array([(x, y) for y in ticks for x in ticks])
    cells = []
    for j in range(cells_per_side):
        for i in range(cells_per_side):
            a = j * (cells_per_side + 1) + i
            diagonal = a + cells_per_side + 2
            cells += [(a, a + 1, diagonal), (a, diagonal, diagonal - 1)]
    return create_triangle_mesh_from_arrays(vertices, np.array(cells))


def _solve(space, problem, exactness_degree: int):
    quadrature = space.element.reference_cell.create_quadrature(exactness_degree)
    basis = create_cell_basis(space, quadrature)
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    local_matrices = integrate_cellwise(
        elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
    )
    local_vectors = integrate_cellwise(
        elliptic_linear_form(basis, coefficients), basis.physical_weights
    )
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        assemble_matrix(local_matrices, space, space),
        assemble_vector(local_vectors, space),
        dirichlet_dofs,
        dirichlet_values,
    )
    free = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full = expand_condensed_solution(free, condensed)
    return FiniteElementFunction(space=space, dof_values=full), basis


def _quadratic(points):
    x, y = points[..., [0]], points[..., [1]]
    return x**2 + 3.0 * x * y - 2.0 * y**2


def _quadratic_gradient(points):
    x, y = points[..., [0]], points[..., [1]]
    return jnp.concatenate((2.0 * x + 3.0 * y, 3.0 * x - 4.0 * y), axis=-1)


@pytest.mark.unit
@pytest.mark.integration
def test_estimator_vanishes_for_exact_quadratic_p2_solution() -> None:
    """P2 reproduces a quadratic solution exactly, so R_K = f + Lap(u_h) and
    every flux jump vanish."""
    mesh = _structured_unit_square(3)
    space = create_finite_element_space(
        mesh, LagrangeTriangleP2(reference_cell=ReferenceTriangle())
    )
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(2.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, _quadratic),
        ),
    )
    solution, basis = _solve(space, problem, exactness_degree=4)
    facet_basis = create_facet_basis(basis, ReferenceInterval().create_quadrature(4))

    cell_indicators, global_estimator = compute_residual_error_estimator(
        solution, basis, facet_basis, problem
    )
    assert cell_indicators.shape == (mesh.cells_to_vertices.shape[0], 1, 1, 1)
    assert float(global_estimator) < 1e-10


@pytest.mark.unit
@pytest.mark.parametrize("element_type", [LagrangeTriangleP1, LagrangeTriangleP2])
def test_flux_divergence_is_exact_for_linear_diffusion(element_type) -> None:
    """For A = a I with linear a, A grad u_h lies in the element space on each
    cell, so div(A grad u_h) = grad a . grad u_h + a Lap(u_h) is reproduced."""
    mesh = _structured_unit_square(4)
    space = create_finite_element_space(
        mesh, element_type(reference_cell=ReferenceTriangle())
    )
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)

    def scalar_diffusion(points):
        return 1.0 + points[..., [0]] + 2.0 * points[..., [1]]

    solution = interpolate_function(_quadratic, space)
    flux_divergence = evaluate_flux_divergence(
        solution, basis, isotropic_tensor_coefficient(scalar_diffusion)
    )

    points = basis.physical_points
    if element_type is LagrangeTriangleP1:
        gradients = gather_vector(solution.dof_values, space).mT @ basis.gradients
        laplacian = 0.0
    else:
        gradients = _quadratic_gradient(points)
        laplacian = 2.0 - 4.0
    expected = (
        jnp.sum(jnp.array([1.0, 2.0]) * gradients, axis=-1, keepdims=True)
        + scalar_diffusion(points) * laplacian
    )
    assert jnp.allclose(
        jnp.broadcast_to(flux_divergence, expected.shape), expected, atol=1e-11
    )


@pytest.mark.unit
def test_flux_divergence_vanishes_for_p1_with_constant_diffusion() -> None:
    mesh = _structured_unit_square(3)
    space = create_finite_element_space(
        mesh, LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    )
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)
    solution = interpolate_function(_quadratic, space)
    diffusion = constant_tensor_coefficient(jnp.array([[2.0, 0.3], [0.3, 1.0]]))
    flux_divergence = evaluate_flux_divergence(solution, basis, diffusion)
    assert jnp.max(jnp.abs(flux_divergence)) < 1e-12


@pytest.mark.unit
def test_facet_jump_indicators_vanish_for_equal_fluxes() -> None:
    mesh = _structured_unit_square(3)
    space = create_finite_element_space(
        mesh, LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    )
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)
    facet_basis = create_facet_basis(basis, ReferenceInterval().create_quadrature(2))
    solution = interpolate_function(_quadratic, space)
    diffusion = constant_tensor_coefficient(jnp.eye(2))
    flux_plus, _ = evaluate_facet_fluxes(solution, facet_basis, diffusion)

    indicators = compute_facet_jump_indicators(facet_basis, flux_plus, flux_plus)
    assert indicators.shape == (mesh.cells_to_vertices.shape[0], 1, 1, 1)
    assert jnp.all(indicators == 0.0)


@pytest.mark.unit
def test_cell_indicators_sum_to_global_squared(
    four_triangle_center_vertex_mesh,
) -> None:
    """eta == sqrt(sum_K eta_K^2), a basic reduce-consistency check, on a
    solution that is *not* exact (so eta_K is genuinely nonzero)."""
    mesh = four_triangle_center_vertex_mesh
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    problem = create_poisson_sin_sin_problem(mesh)

    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
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
    free = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full = expand_condensed_solution(free, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full)

    facet_quadrature = ReferenceInterval().create_quadrature(4)
    facet_basis = create_facet_basis(basis, facet_quadrature)
    cell_indicators, global_estimator = compute_residual_error_estimator(
        solution, basis, facet_basis, problem
    )
    assert jnp.all(cell_indicators >= 0.0)
    assert float(global_estimator) > 0.0
    assert float(global_estimator) == pytest.approx(
        float(jnp.sqrt(jnp.sum(cell_indicators**2))), rel=1e-12
    )


@pytest.mark.unit
@pytest.mark.integration
def test_estimator_bounded_effectivity_for_smooth_problem() -> None:
    """A basic sanity bound: for a smooth problem the estimator should be
    the same order of magnitude as the true H1 error, not wildly off.
    """
    mesh = create_gmsh_unit_square_mesh(8)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    problem = create_poisson_sin_sin_problem(mesh)

    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
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
    free = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full = expand_condensed_solution(free, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full)

    facet_quadrature = ReferenceInterval().create_quadrature(4)
    facet_basis = create_facet_basis(basis, facet_quadrature)
    _, global_estimator = compute_residual_error_estimator(
        solution, basis, facet_basis, problem
    )
    h1_error = float(compute_h1_seminorm_error(solution, basis, problem))
    effectivity_index = float(global_estimator) / h1_error
    assert 1.0 < effectivity_index < 20.0


@pytest.mark.unit
def test_jump_accumulator_matches_hand_derivation(
    two_triangle_unit_square_mesh,
) -> None:
    """Exact hand-derived check of the segment_sum accumulator, using
    non-nodal-affine dof values chosen so grad(u_h) genuinely differs
    between the two cells while u_h itself stays continuous (i.e. a valid,
    if contrived, conforming P1 function) -- isolating the facet-jump term
    from everything else (source = beta = c = 0, so the interior residual
    is identically zero for both cells).

    Global dofs: v0=0, v1=0, v2=0, v3=5 (only the cell-1-exclusive vertex
    nonzero). Cell 0 = (v0,v1,v2) then has u == 0 identically (grad = 0);
    cell 1 = (v0,v2,v3) has u = -5x + 5y (grad = (-5, 5)), continuous with
    cell 0 at the two DOFs they share (v0 = v2 = 0 on both sides). With
    n_e = (-1, 1)/sqrt(2) (pointing cell0 -> cell1, this mesh's known
    orientation) and A = I:
        J_e = (grad|+ - grad|-) . n_e = ((0,0) - (-5,5)) . n_e
            = (5,-5) . (-1,1)/sqrt(2) = -10/sqrt(2) = -5*sqrt(2)
        J_e^2 = 50 (constant along the facet, since both gradients are)
        contribution = 0.5 * h_e * (J_e^2 * h_e) = 0.5 * h_e^2 * J_e^2
                     = 0.5 * 2 * 50 = 50
    landing on *both* cells (interior residual == 0), so eta_K == sqrt(50)
    for both, and the global estimator == sqrt(50 + 50) == 10.
    """
    mesh = two_triangle_unit_square_mesh
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    problem = EllipticProblem(**_ZERO_PROBLEM_KWARGS, dirichlet_conditions=())

    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)
    facet_quadrature = ReferenceInterval().create_quadrature(2)
    facet_basis = create_facet_basis(basis, facet_quadrature)

    assert list(facet_basis.cells_plus.tolist()) == [0]
    assert list(facet_basis.cells_minus.tolist()) == [1]

    dof_values = jnp.array([[0.0], [0.0], [0.0], [5.0]])
    solution = FiniteElementFunction(space=space, dof_values=dof_values)

    cell_indicators, global_estimator = compute_residual_error_estimator(
        solution, basis, facet_basis, problem
    )
    assert cell_indicators.shape == (2, 1, 1, 1)
    assert jnp.allclose(
        jnp.ravel(cell_indicators), jnp.sqrt(jnp.array([50.0, 50.0])), rtol=1e-10
    )
    assert float(global_estimator) == pytest.approx(10.0, rel=1e-10)
