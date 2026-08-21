"""Tests for FiniteElementFunction, interpolation, and evaluation.

See Section 11 of the architecture specification; ERR-03 and ERR-04 of the
FEM testing plan (the latter, higher-order interpolation convergence,
covered by ``test_p2_interpolation_l2_convergence_rate`` below).
"""

import dataclasses
import math

import jax.numpy as jnp
import pytest

from jax_fem.diagnostics import compute_l2_error
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.function import (
    FiniteElementFunction,
    evaluate_finite_element_function,
    interpolate_function,
)
from jax_fem.mesh import create_gmsh_unit_square_mesh, create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.reference_cell.quadrature import QuadratureRule
from jax_fem.space import (
    create_cell_basis,
    create_finite_element_space,
    gather_cell_dof_values,
)

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _make_space(vertices=_UNIT_SQUARE_VERTICES, cells=_UNIT_SQUARE_CELLS):
    mesh = create_triangle_mesh_from_arrays(vertices, cells)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    return mesh, create_finite_element_space(mesh, element)


@pytest.mark.unit
def test_finite_element_function_shape_validation() -> None:
    _, space = _make_space()
    with pytest.raises(ValueError, match="dof_values"):
        FiniteElementFunction(space=space, dof_values=jnp.zeros((3, 1)))


@pytest.mark.unit
def test_gather_cell_dof_values() -> None:
    _, space = _make_space()
    dof_values = jnp.arange(4.0)[:, None]  # (4, 1): value == dof index
    function = FiniteElementFunction(space=space, dof_values=dof_values)
    gathered = gather_cell_dof_values(function)
    assert gathered.shape == (2, 3, 1)
    assert jnp.array_equal(gathered, dof_values[space.cells_to_dofs])


@pytest.mark.unit
def test_interpolate_function_matches_dof_coordinates() -> None:
    _, space = _make_space()

    def u(x):
        return (2.0 * x[..., 0] + 3.0 * x[..., 1] + 1.0)[..., None]

    function = interpolate_function(u, space)
    assert function.dof_values.shape == (4, 1)
    x, y = space.dof_coordinates[:, 0, 0], space.dof_coordinates[:, 0, 1]
    assert jnp.allclose(
        function.dof_values[:, 0], 2.0 * x + 3.0 * y + 1.0, atol=1e-14, rtol=1e-12
    )


@pytest.mark.unit
def test_interpolate_function_non_nodal_element_raises() -> None:
    """interpolate_function is generic over any nodal element (is_nodal ==
    True, Section 3.2): it dispatches on that flag rather than an
    isinstance check, and rejects a non-nodal element (e.g. a future
    H(div) element).
    """
    _, space = _make_space()

    @dataclasses.dataclass(frozen=True, slots=True, eq=False)
    class _FakeNonNodalElement:
        reference_cell: object
        polynomial_degree: int = 1
        # 3 matches cells_to_dofs, unlike LagrangeTriangleP1 for isinstance():
        number_of_local_dofs: int = 3
        value_shape: tuple = (1,)
        entity_dofs: dict = dataclasses.field(
            default_factory=lambda: {0: ((0,), (1,), (2,)), 1: ((), (), ()), 2: ((),)}
        )
        is_nodal: bool = False

        def tabulate_basis_values(self, reference_points):  # noqa: D102
            raise NotImplementedError

        def tabulate_basis_gradients(self, reference_points):  # noqa: D102
            raise NotImplementedError

    fake_space = dataclasses.replace(
        space,
        element=_FakeNonNodalElement(reference_cell=space.element.reference_cell),
    )
    with pytest.raises(NotImplementedError, match="nodal"):
        interpolate_function(constant_scalar_coefficient(0.0), fake_space)


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.property
def test_polynomial_interpolation_exactness_on_physical_mesh(
    four_triangle_center_vertex_mesh, p1_element
) -> None:
    """ERR-03: DOF coordinates, interpolation, mapping, connectivity, and
    physical evaluation, together, for an affine polynomial (the space P1
    exactly represents). Includes points near, but not only on,
    interelement boundaries -- exercising every cell of the mesh, not
    just one in isolation.
    """
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p1_element)

    a, b, c = 2.0, -3.0, 5.0

    def q(x):
        return (a * x[..., 0] + b * x[..., 1] + c)[..., None]

    def grad_q(x):
        return jnp.broadcast_to(jnp.asarray([a, b]), x.shape[:-2] + (1, 2))

    function = interpolate_function(q, space)

    # Points inside each of the 4 cells plus points close to (but not
    # exactly on) the shared interior edges, in barycentric coordinates
    # relative to the reference triangle.
    reference_points = jnp.asarray(
        [
            [
                [[0.2, 0.2]],
                [[0.05, 0.9]],
                [[0.9, 0.05]],
                [[0.49, 0.49]],
                [[0.02, 0.02]],
            ]
        ]
    )
    number_of_points = reference_points.shape[1]
    weights = jnp.full((1, number_of_points, 1, 1), 1.0 / number_of_points)
    quadrature = QuadratureRule(
        reference_cell=space.element.reference_cell,
        points=reference_points,
        weights=weights,
        exactness_degree=1,
    )
    basis = create_cell_basis(space, quadrature)
    evaluation = evaluate_finite_element_function(function, basis)

    assert jnp.allclose(
        evaluation.values, q(basis.physical_points), atol=1e-14, rtol=1e-12
    )
    assert jnp.allclose(
        evaluation.gradients, grad_q(basis.physical_points), atol=1e-14, rtol=1e-12
    )


@pytest.mark.unit
def test_evaluate_finite_element_function_rejects_mismatched_space() -> None:
    _, space_a = _make_space()
    _, space_b = _make_space()  # a distinct FiniteElementSpace instance
    quadrature = space_a.element.reference_cell.create_quadrature(2)
    basis_a = create_cell_basis(space_a, quadrature)

    function_b = interpolate_function(constant_scalar_coefficient(0.0), space_b)
    with pytest.raises(ValueError, match="same FiniteElementSpace"):
        evaluate_finite_element_function(function_b, basis_a)


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.property
def test_p2_polynomial_interpolation_exactness_on_physical_mesh(
    four_triangle_center_vertex_mesh, p2_element
) -> None:
    """ERR-03 for P2: a full quadratic polynomial (P2's local space exactly)
    interpolated on a physical multi-cell mesh, evaluated at non-nodal
    points -- the P2 analogue of the existing P1 affine-polynomial check.
    """
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p2_element)

    a, b, c, d, e, f = 2.0, -3.0, 5.0, 0.7, -1.1, 1.6

    def q(x):
        xx, yy = x[..., 0], x[..., 1]
        return (a * xx + b * yy + c + d * xx**2 + e * xx * yy + f * yy**2)[..., None]

    def grad_q(x):
        xx, yy = x[..., 0], x[..., 1]
        grad_x = a + 2.0 * d * xx + e * yy
        grad_y = b + e * xx + 2.0 * f * yy
        return jnp.stack((grad_x, grad_y), axis=-1)

    function = interpolate_function(q, space)

    reference_points = jnp.asarray(
        [
            [
                [[0.2, 0.2]],
                [[0.05, 0.9]],
                [[0.9, 0.05]],
                [[0.49, 0.49]],
                [[0.02, 0.02]],
            ]
        ]
    )
    number_of_points = reference_points.shape[1]
    weights = jnp.full((1, number_of_points, 1, 1), 1.0 / number_of_points)
    quadrature = QuadratureRule(
        reference_cell=space.element.reference_cell,
        points=reference_points,
        weights=weights,
        exactness_degree=1,
    )
    basis = create_cell_basis(space, quadrature)
    evaluation = evaluate_finite_element_function(function, basis)

    assert jnp.allclose(
        evaluation.values, q(basis.physical_points), atol=1e-11, rtol=1e-10
    )
    assert jnp.allclose(
        evaluation.gradients, grad_q(basis.physical_points), atol=1e-10, rtol=1e-9
    )


@pytest.mark.convergence
@pytest.mark.slow
def test_p2_interpolation_l2_convergence_rate() -> None:
    """ERR-04: pure nodal-interpolation L2 error (no PDE solve) for a smooth
    non-polynomial function converges at P2's approximation-theory rate
    ``O(h^3)``, cheaper to check than the full PDE convergence study
    (``convergence/test_p2_convergence.py``) since it needs no assembly or
    linear solve.
    """

    def u(x):
        xx, yy = x[..., 0], x[..., 1]
        return (jnp.sin(jnp.pi * xx) * jnp.sin(jnp.pi * yy))[..., None]

    # Reuses EllipticProblem purely as a carrier for exact_solution, since
    # compute_l2_error is defined against that protocol; no PDE is solved.
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
        exact_solution=u,
    )

    element = LagrangeTriangleP2(reference_cell=ReferenceTriangle())
    resolutions = (4, 8, 16)
    l2_errors = []
    for n in resolutions:
        mesh = create_gmsh_unit_square_mesh(n)
        space = create_finite_element_space(mesh, element)
        quadrature = space.element.reference_cell.create_quadrature(5)
        basis = create_cell_basis(space, quadrature)
        function = interpolate_function(u, space)
        l2_errors.append(float(compute_l2_error(function, basis, problem)))

    assert all(a > b for a, b in zip(l2_errors, l2_errors[1:], strict=False))
    for error_coarse, error_fine, n_coarse, n_fine in zip(
        l2_errors, l2_errors[1:], resolutions, resolutions[1:], strict=False
    ):
        rate = math.log(error_coarse / error_fine) / math.log(n_fine / n_coarse)
        assert rate == pytest.approx(3.0, abs=0.3)
