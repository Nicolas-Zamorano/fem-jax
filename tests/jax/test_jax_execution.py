"""JAX execution-path tests.

There is no float32 comparison: ``jax_fem/__init__.py`` always enables
``jax_enable_x64`` at import time, so there is no float32 code path.

The library's dataclasses (``CellBasis``, ``FiniteElementSpace``,
``TriangleMesh``, ...) are not registered as JAX pytrees, so they cannot be
traced inputs or outputs of a jitted function. The tests below either pass
them as static arguments, close over them, or compare values extracted from
them, with plain arrays or NamedTuples as the traced inputs.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_matrix, assemble_vector, integrate_cellwise
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction, evaluate_finite_element_function
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


def _make_space_and_basis(exactness_degree: int = 2):
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(exactness_degree)
    basis = create_cell_basis(space, quadrature)
    return space, basis


def _make_coefficients(basis):
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.array([[2.0, 0.3], [0.3, 1.0]])),
        advection=constant_vector_coefficient(jnp.array([0.5, -0.2])),
        reaction=constant_scalar_coefficient(1.5),
        source=constant_scalar_coefficient(3.0),
        dirichlet_conditions=(),
    )
    return evaluate_elliptic_coefficients(problem, basis.physical_points)


@pytest.mark.jax
@pytest.mark.integration
def test_eager_and_jit_agree_for_elliptic_forms() -> None:
    """Elliptic_bilinear_form/elliptic_linear_form under eager vs. jit.

    ``basis`` is a non-pytree dataclass, so it is marked static;
    ``coefficients`` (an ``EllipticCoefficientValues`` NamedTuple, a real
    pytree of arrays) is the actual traced input.
    """
    _, basis = _make_space_and_basis()
    coefficients = _make_coefficients(basis)

    eager_bilinear = elliptic_bilinear_form(basis, basis, coefficients)
    jitted_bilinear = jax.jit(elliptic_bilinear_form, static_argnums=(0, 1))(
        basis, basis, coefficients
    )
    assert jnp.allclose(eager_bilinear, jitted_bilinear, atol=1e-14, rtol=1e-12)

    eager_linear = elliptic_linear_form(basis, coefficients)
    jitted_linear = jax.jit(elliptic_linear_form, static_argnums=(0,))(
        basis, coefficients
    )
    assert jnp.allclose(eager_linear, jitted_linear, atol=1e-14, rtol=1e-12)


@pytest.mark.jax
@pytest.mark.integration
def test_eager_and_jit_agree_for_assembly() -> None:
    """Evaluate, integrate, and assemble under eager execution vs. jit.

    ``basis`` is closed over (a non-pytree dataclass); ``coefficients`` is the
    traced input. The assembled ``BCOO`` matrix is a registered pytree and a
    valid jitted return value.
    """
    _, basis = _make_space_and_basis()
    coefficients = _make_coefficients(basis)

    def assemble_system(coefficients):
        local_matrices = integrate_cellwise(
            elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
        )
        local_vectors = integrate_cellwise(
            elliptic_linear_form(basis, coefficients), basis.physical_weights
        )
        return (
            assemble_matrix(local_matrices, basis.space, basis.space),
            assemble_vector(local_vectors, basis.space),
        )

    eager_matrix, eager_vector = assemble_system(coefficients)
    jitted_matrix, jitted_vector = jax.jit(assemble_system)(coefficients)
    assert jnp.allclose(
        eager_matrix.todense(), jitted_matrix.todense(), atol=1e-14, rtol=1e-12
    )
    assert jnp.allclose(eager_vector, jitted_vector, atol=1e-14, rtol=1e-12)


@pytest.mark.jax
@pytest.mark.integration
def test_eager_and_jit_agree_for_function_evaluation() -> None:
    """Evaluate_finite_element_function under eager vs. jit.

    ``space`` and ``basis`` are static; ``dof_values`` (a plain array) is
    the traced input, exercising real trace-dependent behavior.
    """
    space, basis = _make_space_and_basis()
    dof_values = jnp.array([[1.0], [2.0], [-1.0], [0.5]])

    def evaluate(values):
        function = FiniteElementFunction(space=space, dof_values=values)
        return evaluate_finite_element_function(function, basis)

    eager = evaluate(dof_values)
    jitted = jax.jit(evaluate)(dof_values)

    assert jnp.allclose(eager.values, jitted.values, atol=1e-14, rtol=1e-12)
    assert jnp.allclose(eager.gradients, jitted.gradients, atol=1e-14, rtol=1e-12)


@pytest.mark.jax
@pytest.mark.integration
def test_python_loop_matches_batched_local_stiffness_assembly() -> None:
    """Batched (broadcast-vectorized) vs. looped results agree.

    This library batches over cells via direct broadcasting rather than
    ``jax.vmap``, so this compares the
    library's batched per-cell local matrices against an independent
    Python loop that builds one cell's CellBasis and local matrix at a
    time, on a small heterogeneous set of affine triangles.
    """
    vertices = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.3, 1.7),
        (2.1, 0.4),
    )
    cells = ((0, 1, 2), (0, 2, 3), (1, 4, 2), (0, 5, 1))
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())

    mesh = create_triangle_mesh_from_arrays(vertices, cells)
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(3)
    basis = create_cell_basis(space, quadrature)
    coefficients = _make_coefficients(basis)
    batched_local_matrices = integrate_cellwise(
        elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
    )

    looped_local_matrices = []
    for cell in cells:
        one_cell_mesh = create_triangle_mesh_from_arrays(vertices, (cell,))
        one_space = create_finite_element_space(one_cell_mesh, element)
        one_quadrature = one_space.element.reference_cell.create_quadrature(3)
        one_basis = create_cell_basis(one_space, one_quadrature)
        one_coefficients = _make_coefficients(one_basis)
        one_pointwise = elliptic_bilinear_form(one_basis, one_basis, one_coefficients)
        one_local = integrate_cellwise(one_pointwise, one_basis.physical_weights)[0]
        looped_local_matrices.append(one_local)
    looped_local_matrices = jnp.stack(looped_local_matrices)

    assert batched_local_matrices.shape == looped_local_matrices.shape
    assert jnp.allclose(
        batched_local_matrices, looped_local_matrices, atol=1e-14, rtol=1e-12
    )


@pytest.mark.jax
def test_autodiff_of_assembled_matrix_matches_finite_difference() -> None:
    """Assembly is differentiable w.r.t. a coefficient parameter.

    Differentiates a scalar reduction of the assembled stiffness matrix
    with respect to a diffusion scale ``alpha`` (``A = alpha * I``) and
    compares ``jax.grad`` against a central finite difference.
    """
    _, basis = _make_space_and_basis()

    def total_stiffness(alpha: jax.Array) -> jax.Array:
        problem = EllipticProblem(
            diffusion=constant_tensor_coefficient(alpha * jnp.eye(2)),
            advection=constant_vector_coefficient(jnp.zeros(2)),
            reaction=constant_scalar_coefficient(0.0),
            source=constant_scalar_coefficient(0.0),
            dirichlet_conditions=(),
        )
        coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
        local_matrices = integrate_cellwise(
            elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
        )
        matrix = assemble_matrix(local_matrices, basis.space, basis.space)
        return jnp.sum(matrix.todense())

    alpha0 = 1.7
    autodiff_derivative = jax.grad(total_stiffness)(alpha0)

    epsilon = 1e-4
    finite_difference_derivative = (
        total_stiffness(alpha0 + epsilon) - total_stiffness(alpha0 - epsilon)
    ) / (2.0 * epsilon)

    assert autodiff_derivative == pytest.approx(
        float(finite_difference_derivative), rel=1e-5
    )
