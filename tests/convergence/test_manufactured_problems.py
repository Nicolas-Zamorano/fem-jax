"""Tests for the manufactured problems in ``problem/manufactured.py``.

Verification happens at two independent levels:

1. Continuous, autodiff-based: ``exact_gradient`` is checked against
   ``jax.grad`` of ``exact_solution``, and ``source`` is checked against the
   strong-form residual ``-div(A grad u) + beta . grad u + c u`` computed via
   ``jax.jacobian``/``jax.hessian`` of the coefficient and solution
   functions. This never touches the mesh/FEM machinery, so it isolates
   translation mistakes in the manufactured math itself.
2. Discrete: one coarse FEM solve per problem, checking the computed L2/H1
   errors are small (and, for the smooth problems, converge at the expected
   P1 rate -- see ``test_elliptic_convergence.py`` for the full-rate study).

ELL-04 of the FEM testing plan (variable-coefficient manufactured problem)
is covered by ``SimpleEllipticProblem`` here.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import compute_h1_seminorm_error, compute_l2_error
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_structured_unit_square_mesh
from jax_fem.problem import (
    EllipticProblem,
    create_poisson_cos_sin_problem,
    create_poisson_exponential_problem,
    create_poisson_sin_sin_problem,
    create_poisson_singular_problem,
    create_simple_elliptic_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

# Interior sample points, away from the domain boundary and (for the
# singular problem) away from the origin corner.
_SAMPLE_POINTS = jnp.array(
    [
        (0.12, 0.31),
        (0.34, 0.71),
        (0.63, 0.44),
        (0.81, 0.19),
        (0.5, 0.5),
        (0.05, 0.6),
        (0.92, 0.92),
    ]
)

_MESH = create_structured_unit_square_mesh(1)  # only used as a mesh handle

_PROBLEM_FACTORIES = {
    "sin_sin": create_poisson_sin_sin_problem,
    "cos_sin": create_poisson_cos_sin_problem,
    "exponential": create_poisson_exponential_problem,
    "singular": create_poisson_singular_problem,
    "simple": create_simple_elliptic_problem,
}


def _as_batch(point: jax.Array) -> jax.Array:
    return point.reshape(1, 1, 2)


def _check_gradient_matches_autodiff(
    problem: EllipticProblem, points: jax.Array
) -> None:
    def u_scalar(point: jax.Array) -> jax.Array:
        return problem.exact_solution(_as_batch(point))[0, 0, 0]

    for point in points:
        autodiff_gradient = jax.grad(u_scalar)(point)
        hand_gradient = problem.exact_gradient(_as_batch(point))[0, 0]
        assert jnp.allclose(autodiff_gradient, hand_gradient, atol=1e-6), point


def _strong_form_residual(problem: EllipticProblem, point: jax.Array) -> jax.Array:
    """``-div(A grad u) + beta . grad u + c u`` at one point, via autodiff."""

    def u_scalar(p: jax.Array) -> jax.Array:
        return problem.exact_solution(_as_batch(p))[0, 0, 0]

    def a_matrix(p: jax.Array) -> jax.Array:
        return problem.diffusion(_as_batch(p))[0]  # (2, 2)

    def beta_vector(p: jax.Array) -> jax.Array:
        return problem.advection(_as_batch(p))[0, 0]  # (2,)

    def c_scalar(p: jax.Array) -> jax.Array:
        return problem.reaction(_as_batch(p))[0, 0, 0]

    grad_u = jax.grad(u_scalar)(point)  # (2,)
    hess_u = jax.hessian(u_scalar)(point)  # (2, 2)
    jac_a = jax.jacobian(a_matrix)(point)  # (2, 2, 2): d A_ij / d x_k

    # div(A grad u) = sum_ij d(A_ij du/dx_j)/dx_i
    #              = sum_ij [dA_ij/dx_i * du/dx_j + A_ij * d2u/(dx_i dx_j)]
    divergence_term = jnp.einsum("iji,j->", jac_a, grad_u) + jnp.einsum(
        "ij,ij->", a_matrix(point), hess_u
    )
    advection_term = jnp.dot(beta_vector(point), grad_u)
    reaction_term = c_scalar(point) * u_scalar(point)

    return -divergence_term + advection_term + reaction_term


def _check_source_matches_strong_form(
    problem: EllipticProblem, points: jax.Array
) -> None:
    for point in points:
        autodiff_source = _strong_form_residual(problem, point)
        hand_source = problem.source(_as_batch(point))[0, 0, 0]
        assert autodiff_source == pytest.approx(float(hand_source), abs=1e-5), point


@pytest.mark.unit
@pytest.mark.jax
@pytest.mark.parametrize("name", _PROBLEM_FACTORIES)
def test_exact_gradient_matches_autodiff(name: str) -> None:
    problem = _PROBLEM_FACTORIES[name](_MESH)
    _check_gradient_matches_autodiff(problem, _SAMPLE_POINTS)


@pytest.mark.unit
@pytest.mark.jax
@pytest.mark.parametrize("name", _PROBLEM_FACTORIES)
def test_source_matches_strong_form_residual(name: str) -> None:
    problem = _PROBLEM_FACTORIES[name](_MESH)
    _check_source_matches_strong_form(problem, _SAMPLE_POINTS)


@pytest.mark.unit
@pytest.mark.parametrize("name", _PROBLEM_FACTORIES)
def test_dirichlet_values_equal_exact_solution_at_boundary(name: str) -> None:
    problem = _PROBLEM_FACTORIES[name](_MESH)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(_MESH, element)
    dofs, values = evaluate_dirichlet_dof_values(problem, space)
    coordinates = space.dof_coordinates[dofs]
    expected = problem.exact_solution(coordinates)[:, :, 0]
    assert jnp.allclose(values, expected, atol=1e-10)


def _solve(problem: EllipticProblem, space, basis) -> FiniteElementFunction:
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full_dof_values = expand_condensed_solution(free_dof_values, condensed)
    return FiniteElementFunction(space=space, dof_values=full_dof_values)


@pytest.mark.integration
@pytest.mark.end_to_end
@pytest.mark.parametrize(
    # Thresholds are loose pipeline sanity checks at a single coarse
    # resolution (n=16), not tight accuracy targets or rate assertions --
    # the actual P1 convergence rate is verified in test_elliptic_convergence.py.
    # Calibrated with ~1.5-2x margin above observed n=16 errors.
    ("name", "l2_tolerance", "h1_tolerance"),
    [
        ("sin_sin", 1e-2, 0.3),
        ("cos_sin", 1e-2, 0.3),
        ("exponential", 5e-2, 2.0),
        ("simple", 6e-2, 2.0),
        ("singular", 1e-2, 0.1),
    ],
)
def test_discrete_solve_error_is_small(
    name: str, l2_tolerance: float, h1_tolerance: float
) -> None:
    mesh = create_structured_unit_square_mesh(16)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(5)
    basis = create_cell_basis(space, quadrature)

    problem = _PROBLEM_FACTORIES[name](mesh)
    solution = _solve(problem, space, basis)

    l2_error = float(compute_l2_error(solution, basis, problem))
    h1_error = float(compute_h1_seminorm_error(solution, basis, problem))
    assert l2_error < l2_tolerance, l2_error
    assert h1_error < h1_tolerance, h1_error
