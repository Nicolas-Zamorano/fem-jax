"""Tests for the classical residual-based a posteriori error estimator.

See Section 4 of the implementation plan (``.context/implementation_plan.md``).
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import compute_h1_seminorm_error, compute_residual_error_estimator
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_gmsh_unit_square_mesh, create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
    create_poisson_sin_sin_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from jax_fem.space.facet_basis import create_facet_basis

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))

_ZERO_PROBLEM_KWARGS = dict(
    diffusion=constant_tensor_coefficient(jnp.eye(2)),
    advection=constant_vector_coefficient(jnp.zeros(2)),
    reaction=constant_scalar_coefficient(0.0),
    source=constant_scalar_coefficient(0.0),
)


def _estimate(solution, space, problem, cell_degree: int = 4, facet_degree: int = 4):
    quadrature = space.element.reference_cell.create_quadrature(cell_degree)
    basis = create_cell_basis(space, quadrature)
    facet_quadrature = ReferenceInterval().create_quadrature(facet_degree)
    facet_basis = create_facet_basis(basis, facet_quadrature)
    return compute_residual_error_estimator(solution, basis, facet_basis, problem), basis


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
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
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


@pytest.mark.unit
def test_estimator_rejects_p2() -> None:
    """The interior-residual term assumes div(A grad(u_h)) == 0 (exact only
    for a constant-reference-gradient element); P2's gradients vary, so it
    must raise rather than silently compute a wrong residual term.
    """
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP2(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    problem = EllipticProblem(
        **_ZERO_PROBLEM_KWARGS,
        dirichlet_conditions=(),
    )
    solution = FiniteElementFunction(
        space=space, dof_values=jnp.zeros((space.number_of_dofs, 1))
    )
    with pytest.raises(NotImplementedError, match="LagrangeTriangleP1"):
        _estimate(solution, space, problem)


@pytest.mark.unit
def test_cell_indicators_sum_to_global_squared(four_triangle_center_vertex_mesh) -> None:
    """eta == sqrt(sum_K eta_K^2), a basic reduce-consistency check, on a
    solution that is *not* exact (so eta_K is genuinely nonzero)."""
    mesh = four_triangle_center_vertex_mesh
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    problem = create_poisson_sin_sin_problem(mesh)

    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
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
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
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
def test_jump_accumulator_matches_hand_derivation(two_triangle_unit_square_mesh) -> None:
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
    assert jnp.allclose(cell_indicators, jnp.sqrt(jnp.array([50.0, 50.0])), rtol=1e-10)
    assert float(global_estimator) == pytest.approx(10.0, rel=1e-10)
