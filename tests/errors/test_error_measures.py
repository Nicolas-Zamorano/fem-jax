"""Tests for exact-error functionals and norms.

See Sections 13.3 and 18.3 of the architecture specification; ERR-01 and
ERR-02 of the FEM testing plan.
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import integrate_cellwise
from jax_fem.diagnostics import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_l2_error,
    compute_relative_l2_error,
    h1_seminorm_error_density,
    l2_error_density,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.function import evaluate_finite_element_function, interpolate_function
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _make_basis_and_problem(
    exact_solution=None, exact_gradient=None, advection_divergence=None
):
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.array([[2.0, 0.0], [0.0, 3.0]])),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
        exact_solution=exact_solution,
        exact_gradient=exact_gradient,
        advection_divergence=advection_divergence,
    )
    return space, basis, problem


def _affine_u(a: float, b: float, c: float):
    def u(x):
        return (a * x[..., 0] + b * x[..., 1] + c)[..., None]

    return u


def _affine_grad(a: float, b: float):
    def grad(x):
        return jnp.broadcast_to(jnp.asarray((a, b)), x.shape[:-2] + (1, 2))

    return grad


@pytest.mark.unit
def test_errors_vanish_when_solution_equals_exact_affine_function() -> None:
    """Comparing a function with itself returns zero error (part of ERR-01)."""
    a, b, c = 2.0, -1.0, 3.0
    u_exact = _affine_u(a, b, c)
    grad_exact = _affine_grad(a, b)
    space, basis, problem = _make_basis_and_problem(
        u_exact, grad_exact, constant_scalar_coefficient(0.0)
    )

    solution = interpolate_function(u_exact, space)  # P1 represents affine u exactly

    assert compute_l2_error(solution, basis, problem) == pytest.approx(0.0, abs=1e-10)
    assert compute_relative_l2_error(solution, basis, problem) == pytest.approx(
        0.0, abs=1e-10
    )
    assert compute_h1_seminorm_error(solution, basis, problem) == pytest.approx(
        0.0, abs=1e-10
    )
    assert compute_energy_error(solution, basis, problem) == pytest.approx(
        0.0, abs=1e-10
    )


@pytest.mark.unit
def test_errors_are_true_scalars() -> None:
    u_exact = _affine_u(1.0, 1.0, 0.0)
    grad_exact = _affine_grad(1.0, 1.0)
    space, basis, problem = _make_basis_and_problem(
        u_exact, grad_exact, constant_scalar_coefficient(0.0)
    )
    solution = interpolate_function(_affine_u(0.0, 0.0, 0.0), space)  # u_h = 0

    assert compute_l2_error(solution, basis, problem).shape == ()
    assert compute_h1_seminorm_error(solution, basis, problem).shape == ()
    assert compute_energy_error(solution, basis, problem).shape == ()


@pytest.mark.unit
def test_l2_norm_of_constant_one_on_unit_square() -> None:
    """ERR-01: ||1||_L2 == 1 on the unit square, independent of any PDE solve.

    Computed as ||0 - 1||_L2 via compute_l2_error with u_h = 0.
    """
    u_exact = _affine_u(0.0, 0.0, 1.0)
    grad_exact = _affine_grad(0.0, 0.0)
    space, basis, problem = _make_basis_and_problem(
        u_exact, grad_exact, constant_scalar_coefficient(0.0)
    )
    solution = interpolate_function(_affine_u(0.0, 0.0, 0.0), space)

    assert compute_l2_error(solution, basis, problem) == pytest.approx(1.0)
    assert compute_relative_l2_error(solution, basis, problem) == pytest.approx(1.0)


@pytest.mark.unit
def test_h1_seminorm_of_x_plus_2y_on_unit_square() -> None:
    """ERR-01: |x + 2y|_H1 == sqrt(5) on the unit square.

    grad(x + 2y) = (1, 2), constant, so |x+2y|_H1^2 = (1^2+2^2) * area = 5.
    Computed as |0 - (x+2y)|_H1 via compute_h1_seminorm_error with u_h = 0.
    """
    u_exact = _affine_u(1.0, 2.0, 0.0)
    grad_exact = _affine_grad(1.0, 2.0)
    space, basis, problem = _make_basis_and_problem(
        u_exact, grad_exact, constant_scalar_coefficient(0.0)
    )
    solution = interpolate_function(_affine_u(0.0, 0.0, 0.0), space)

    assert compute_h1_seminorm_error(solution, basis, problem) == pytest.approx(
        5.0**0.5
    )


@pytest.mark.unit
@pytest.mark.property
def test_cellwise_and_global_error_consistency() -> None:
    """ERR-02: sum of cellwise squared errors equals the square of the global error.

    Detects accidental summation of norms (rather than squared norms).
    """
    space, basis, problem = _make_basis_and_problem(
        _affine_u(1.0, -2.0, 0.5), _affine_grad(1.0, -2.0)
    )
    solution = interpolate_function(_affine_u(0.0, 0.0, 0.0), space)  # nonzero error
    fields = {"solution": evaluate_finite_element_function(solution, basis)}

    l2_density = l2_error_density(basis, fields, problem)
    cellwise_l2_squared = integrate_cellwise(l2_density, basis.physical_weights)
    global_l2 = compute_l2_error(solution, basis, problem)
    assert float(jnp.sum(cellwise_l2_squared)) == pytest.approx(
        float(global_l2) ** 2, rel=1e-10
    )

    h1_density = h1_seminorm_error_density(basis, fields, problem)
    cellwise_h1_squared = integrate_cellwise(h1_density, basis.physical_weights)
    global_h1 = compute_h1_seminorm_error(solution, basis, problem)
    assert float(jnp.sum(cellwise_h1_squared)) == pytest.approx(
        float(global_h1) ** 2, rel=1e-10
    )


@pytest.mark.unit
def test_l2_error_without_exact_solution_raises() -> None:
    space, basis, problem = _make_basis_and_problem(exact_solution=None)
    solution = interpolate_function(_affine_u(0.0, 0.0, 0.0), space)
    with pytest.raises(ValueError, match="exact_solution"):
        compute_l2_error(solution, basis, problem)


@pytest.mark.unit
def test_h1_error_without_exact_gradient_raises() -> None:
    space, basis, problem = _make_basis_and_problem(
        exact_solution=_affine_u(0.0, 0.0, 0.0), exact_gradient=None
    )
    solution = interpolate_function(_affine_u(0.0, 0.0, 0.0), space)
    with pytest.raises(ValueError, match="exact_gradient"):
        compute_h1_seminorm_error(solution, basis, problem)


@pytest.mark.unit
def test_energy_error_without_advection_divergence_raises() -> None:
    space, basis, problem = _make_basis_and_problem(
        exact_solution=_affine_u(0.0, 0.0, 0.0),
        exact_gradient=_affine_grad(0.0, 0.0),
        advection_divergence=None,
    )
    solution = interpolate_function(_affine_u(0.0, 0.0, 0.0), space)
    with pytest.raises(ValueError, match="advection_divergence"):
        compute_energy_error(solution, basis, problem)
